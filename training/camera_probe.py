"""One-shot camera probe + YOLO sanity image, run once before training starts.

Stages, in order: env sensor config, wide debug view, reset + cached YOLO read,
snapshot export (_save_camera_snapshot), detector gate (_assert_yolo_detection).

Reads the env's own wrist camera and YOLO cache right after reset (exactly what the
policy observes), saves an RGB/depth snapshot, a YOLO-annotated image and a debug
wide view, and aborts training when the camera is blank or nothing is detected.
"""

import json
import os
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg

from config import CAMERA_HEIGHT, CAMERA_WIDTH, YOLO_CACHE_NORM_DEPTH_M, YOLO_CACHE_NORM_PX
from evaluation.camera_capture import draw_cached_detection
from rl_env.camera_utils import configure_camera_sensor, training_rgb_uint8
from rl_env.perception.yolo_cache import _get_yolo_detection_cache


def _configure_env_camera_sensor(env_cfg, args_cli) -> None:
    width, height = configure_camera_sensor(
        env_cfg, args_cli.camera_resolution, args_cli.camera_update_period, "Env"
    )
    if width * height >= CAMERA_WIDTH * CAMERA_HEIGHT and args_cli.num_envs > 64:
        print(
            "[WARNING] High num_envs with 640-class RGB+depth+YOLO can still exhaust RTX resources; "
            "set CAMERA_DEFAULTS['update_period'] or CAMERA_DEFAULTS['resolution'] in config/perception.py after recalibrating YOLO detection."
        )


def _create_wide_camera(scene_env, width: int, height: int):
    """Debug far view of robot + object (scene_wide.png only; destroyed after the probe)."""
    try:
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.sensors.camera", True)
        from isaacsim.sensors.camera import Camera
        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        robot_root = scene_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        eye = torch.tensor([[robot_root[0] - 4.5, robot_root[1] + 4.5, robot_root[2] + 2.0]], dtype=torch.float32)
        target = torch.tensor([[robot_root[0], robot_root[1], robot_root[2] + 0.35]], dtype=torch.float32)
        quat = quat_from_matrix(create_rotation_matrix_from_view(eye, target, up_axis="Z"))[0].cpu().tolist()
        sim_utils.create_prim(
            "/World/debug_far_camera",
            "Camera",
            translation=tuple(float(v) for v in eye[0].tolist()),
            orientation=tuple(float(v) for v in quat),
        )
        far_cam = Camera(prim_path="/World/debug_far_camera", resolution=(width, height))
        far_cam.initialize()
        far_cam.add_rgb_to_frame()
        return far_cam
    except Exception as e:
        print(f"[CAMERA] Wide view setup failed: {e}")
        return None


def _print_intrinsics(wrist_cam, width: int, height: int) -> None:
    """TiledCamera intrinsics, so the YOLO dataset generator can be calibrated to match."""
    try:
        k0 = wrist_cam.data.intrinsic_matrices[0]
        fx, fy = float(k0[0, 0]), float(k0[1, 1])
        cx, cy = float(k0[0, 2]), float(k0[1, 2])
        fov_h = 2 * np.degrees(np.arctan(width / (2 * fx)))
        fov_v = 2 * np.degrees(np.arctan(height / (2 * fy)))
        focal_mm_ref = 1.93
        print(
            f"[CAMERA] TiledCamera intrinsics env_0: "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  "
            f"FoV_H={fov_h:.1f}° FoV_V={fov_v:.1f}°"
        )
        print(
            f"[CAMERA] Dataset generator: use --focal_mm 1.93 "
            f"--h_aperture_mm {focal_mm_ref * width / fx:.4f} --v_aperture_mm {focal_mm_ref * height / fy:.4f}"
        )
    except Exception as e:
        print(f"[CAMERA] Could not read TiledCamera intrinsics: {e}")


def setup_training_camera(log_dir: str, env, args_cli, simulation_app) -> None:
    """Probe the env wrist camera + YOLO cache; training does not start if either fails."""
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("cv2 is unavailable; camera images cannot be saved") from e

    scene_env = env.unwrapped if hasattr(env, "unwrapped") else env
    wrist_cam = scene_env.scene["wrist_camera"]
    width, height = int(wrist_cam.cfg.width), int(wrist_cam.cfg.height)
    output_dir = args_cli.camera_output_dir or os.path.join(log_dir, "camera_probe")
    os.makedirs(output_dir, exist_ok=True)
    wide_path = os.path.join(output_dir, "scene_wide.png")

    far_cam = _create_wide_camera(scene_env, width, height)
    _print_intrinsics(wrist_cam, width, height)

    print(f"[CAMERA] Reading wrist_camera env_0 @ {width}x{height} after reset...")
    env.reset()
    # RTX needs a few rendered frames after reset before the wrist view is current:
    # re-render, re-read the sensor and refill the cache until YOLO sees the object.
    deadline = time.time() + float(args_cli.camera_timeout_sec)
    while True:
        scene_env.sim.render()
        wrist_cam.update(scene_env.step_dt, force_recompute=True)
        scene_env._yolo_detection_cache = {}
        cache = _get_yolo_detection_cache(scene_env, SceneEntityCfg("wrist_camera"))
        if float(cache["has_det"][0]) > 0.5 or time.time() > deadline:
            break
    rgb = wrist_cam.data.output["rgb"][0].detach().cpu().numpy()
    depth = wrist_cam.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
    if rgb.size == 0 or int(rgb.max()) == 0:
        raise RuntimeError("The wrist camera produced no frame after reset; training cancelled.")

    wide_rgb = None
    if far_cam is not None:
        wide_rgb = _wait_for_wide_camera(far_cam, simulation_app, args_cli.camera_timeout_sec)
        try:
            far_cam.destroy()
        except Exception as e:
            print(f"[CAMERA] Wide view camera could not be removed: {e}")

    rgb_u8, rgb_path = _save_camera_snapshot(rgb, depth, wide_rgb, output_dir, wide_path, cv2)
    _assert_yolo_detection(rgb_u8, cache, rgb_path, output_dir, args_cli, cv2)


def _wait_for_wide_camera(cam, simulation_app, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        simulation_app.update()
        rgb = cam.get_rgb()
        if rgb is None:
            rgb = cam.get_rgba()
        if rgb is not None and np.size(rgb) > 0:
            return rgb
        time.sleep(0.05)
    return None


def _save_camera_snapshot(rgb, depth, wide_rgb, output_dir, wide_path, cv2):
    rgb_u8 = training_rgb_uint8(rgb)
    if rgb_u8 is None or rgb_u8.size == 0:
        raise RuntimeError("Could not read RGB from the camera; training cancelled.")

    rgb_path = os.path.join(output_dir, "camera_rgb.png")
    depth_path = os.path.join(output_dir, "camera_depth.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    center_depth = 0.0
    if depth is not None and np.size(depth) > 0:
        depth_arr = np.asarray(depth).astype(np.float32)
        if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[:, :, 0]
        h, w = depth_arr.shape[:2]
        center_depth = float(depth_arr[h // 2, w // 2])
        finite_mask = np.isfinite(depth_arr) & (depth_arr > 0.0)
        if np.any(finite_mask):
            finite_depth = depth_arr[finite_mask]
            d_min = float(np.percentile(finite_depth, 2.0))
            d_max = float(np.percentile(finite_depth, 98.0))
            if d_max - d_min > 1e-6:
                clipped = np.clip(depth_arr, d_min, d_max)
                norm = ((clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                norm[~finite_mask] = 0
            else:
                norm = np.zeros_like(depth_arr, dtype=np.uint8)
        else:
            norm = np.zeros_like(depth_arr, dtype=np.uint8)
        depth_vis = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        cv2.imwrite(depth_path, depth_vis)

    wide_u8 = training_rgb_uint8(wide_rgb) if wide_rgb is not None else None
    if wide_u8 is not None and wide_u8.size > 0 and int(wide_u8.max()) > 0:
        cv2.imwrite(wide_path, cv2.cvtColor(wide_u8, cv2.COLOR_RGB2BGR))
        print(f"[CAMERA] Wide view saved: {wide_path}")
    else:
        print("[CAMERA] Wide view is empty; the debug camera has not produced a valid frame.")

    print(
        f"[CAMERA] ✔ Ready | center_depth={center_depth:.4f}m | "
        f"rgb_saved=True | depth_saved={depth is not None}"
    )

    return rgb_u8, rgb_path


def _assert_yolo_detection(rgb_u8, cache, rgb_path, output_dir, args_cli, cv2):
    detections = []
    if float(cache["has_det"][0]) > 0.5:
        cx, cy = (cache["center_norm"][0] * YOLO_CACHE_NORM_PX).tolist()
        w, h = (cache["size_norm"][0] * YOLO_CACHE_NORM_PX).tolist()
        detections = [{
            "confidence": float(cache["conf"][0]),
            "bbox_xyxy": [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
            "bbox_center_px": [cx, cy],
            "bbox_size_px": [w, h],
            "depth_m": float(cache["depth_norm"][0, 0]) * YOLO_CACHE_NORM_DEPTH_M,
        }]

    ann_path = os.path.join(output_dir, "yolo_annotated.png")
    json_path = os.path.join(output_dir, "yolo_detections.json")
    cv2.imwrite(ann_path, cv2.cvtColor(draw_cached_detection(rgb_u8, cache), cv2.COLOR_RGB2BGR))

    payload = {
        "ok": True,
        "source": "env YOLO cache (same values as the policy observation)",
        "model_path": args_cli.yolo_model_path,
        "image_size": {"width": int(rgb_u8.shape[1]), "height": int(rgb_u8.shape[0])},
        "count": len(detections),
        "detections": detections,
        "raw_path": rgb_path,
        "annotated_path": ann_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[YOLO] Detection saved: {ann_path} | {json_path}")
    if not detections:
        raise RuntimeError(
            "The YOLO startup probe detected no object in the wrist camera; "
            f"see {ann_path}. Training cancelled so the policy does not learn from empty observations."
        )
