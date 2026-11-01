"""
Play a trained UR10e policy in Isaac Sim (one env, video output).

Usage:
    ./isaac-ctl.sh run_isaaclab play.py \
        --checkpoint /divc_ppo/logs/sb3/Isaac-UR10e-DivC-Reach-v0/<run>/model.zip

    ./isaac-ctl.sh run_isaaclab play.py \
        --checkpoint /divc_ppo/logs/sb3/Isaac-UR10e-DivC-Reach-v0/<run>/model.zip \
        --num_episodes 3 --video_length 300

    # Tanpa checkpoint → cari model.zip terbaru:
    ./isaac-ctl.sh run_isaaclab play.py

Output:
    /divc_ppo/play_outputs/<checkpoint_tag>_<timestamp>/
        eval_<timestamp>.mp4   ← viewport video (if rendering is available)
        camera_<timestamp>.mp4 ← wrist-camera video
        camera_det_<timestamp>.mp4 ← video wrist camera + deteksi YOLO
        eval_summary.txt       ← ringkasan reward & sukses
        training_plots/        ← grafik performa training (paper-standard PNG+PDF)
            training_overview.png/pdf   ← dashboard 2×3 metrik kunci
            fig_reward / fig_task_error / fig_detection_rate .png/pdf
            training_keypoints.txt      ← poin penting (final/best per metrik)
"""

# ─── 1. Parse argumen SEBELUM import Isaac Sim ────────────────────────────────
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Import rl_env from this project, not from shared storage
for _m in list(sys.modules):
    if _m == "rl_env" or _m.startswith("rl_env."):
        del sys.modules[_m]


from rl_env.project_config import (
    ASSETS_ROOT,
    PLAY_ENV_ID,
    VIDEO_ROOT,
    log_base,
)
from config import (
    PLAY_DEFAULTS,
    TRAIN_DEFAULTS,
)
from evaluation.inference_bootstrap import (
    apply_runtime_defaults, load_checkpoint_context, apply_checkpoint_model,
    configure_inference_device, configure_inference_detector, load_inference_policy,
)


parser = argparse.ArgumentParser(description="Play a trained UR10e policy with video output.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Override the PLAY_DEFAULTS checkpoint. If both are empty, use the latest model.zip.",
)

AppLauncher.add_app_launcher_args(parser)

# ── Inject headless+cameras SEBELUM parse_args() ──────────────────────────────
# AppLauncher butuh KEDUANYA: headless=True DAN enable_cameras=True
# so that _offscreen_render=True -> PARTIAL_RENDERING -> rgb_array can render.
if not bool(PLAY_DEFAULTS["no_video"]):
    os.environ.setdefault("ENABLE_CAMERAS", "1")
    os.environ.setdefault("HEADLESS", "1")
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")

args_cli = parser.parse_args()

# config.training selected/restored this exact path before perception/task froze.
args_cli.checkpoint = PLAY_DEFAULTS["checkpoint"]
args_cli.num_episodes = int(PLAY_DEFAULTS["num_episodes"])
args_cli.video_length = int(PLAY_DEFAULTS["video_length"])
from config.env_flags import env_int
args_cli.seed = env_int("SEED", int(PLAY_DEFAULTS["seed"]))
args_cli.no_video = bool(PLAY_DEFAULTS["no_video"])
args_cli.no_env_video = bool(PLAY_DEFAULTS["no_env_video"])
args_cli.no_camera_det = bool(PLAY_DEFAULTS["no_camera_det"])
args_cli.real_time = bool(PLAY_DEFAULTS["real_time"])
apply_runtime_defaults(args_cli, PLAY_DEFAULTS)
print("[SETTINGS] Play defaults loaded from config/training.py", flush=True)

_candidate_checkpoint = args_cli.checkpoint
_checkpoint_context = load_checkpoint_context(_candidate_checkpoint)
_checkpoint_agent_cfg_early = _checkpoint_context.agent
vecnorm_path = _checkpoint_context.normalizer
apply_checkpoint_model(args_cli, _checkpoint_context, TRAIN_DEFAULTS)

print(
    f"[MODE] checkpoint PPO_head={args_cli.ppo_head} | "
    f"asym_critic={args_cli.asymmetric_critic}",
    flush=True,
)
_RUN_DEVICE = configure_inference_device(args_cli)
print(
    f"[MODE] play control_mode={args_cli.mode} | PPO_head={args_cli.ppo_head} | "
    f"asym_critic={args_cli.asymmetric_critic} | "
    f"actor_depth_enabled={args_cli.actor_depth_enabled}",
    flush=True,
)

configure_inference_detector(args_cli)
print(
    f"[GPU] play device={args_cli.device} | "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}",
    flush=True,
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── 2. Imports after Isaac Sim starts ────────────────────────────────────────
import time
import torch
import numpy as np
from datetime import datetime

# ─── 3. Register custom task ─────────────────────────────────────────────────
import rl_env  # noqa: F401
from rl_env.env import (
    UR10eEnvCfg_PLAY,
    GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M,
    GT_SUCCESS_XY_TOL_M,
    SCRIPTED_STAY_STEPS_REQUIRED,
    SCRIPTED_STAY_Z_TOL_M,
)
from rl_env.perception.yolo_cache import _get_yolo_detection_cache
from rl_env.env_ids import find_body_id
from rl_env.rewards.error_sources import gt_target_error_w
from rl_env.perception.yolo_runtime import _draw_gripper_exclusion
from rl_env.policy import AsymmetricActorCriticPolicy  # noqa: F401
from rl_env.env.factory import make_isaac_env
from isaaclab.managers import SceneEntityCfg

ENV_ID     = PLAY_ENV_ID
LOG_BASE   = log_base()
VIDEO_BASE = VIDEO_ROOT
os.makedirs(VIDEO_BASE, exist_ok=True)



# ─── 4. Resolve checkpoint ───────────────────────────────────────────────────
checkpoint_path = _candidate_checkpoint
if checkpoint_path is None:
    print(f"[ERROR] No model.zip in {LOG_BASE}.", flush=True)
    simulation_app.close()
    sys.exit(1)

checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_tag = os.path.basename(checkpoint_dir)

print(f"[INFO] Checkpoint : {checkpoint_path}", flush=True)
print(f"[INFO] VecNorm    : {'OK' if os.path.exists(vecnorm_path) else 'not found'}", flush=True)

# ─── 5. Output dir ───────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
_play_output = os.path.join(VIDEO_BASE, f"{checkpoint_tag}_{timestamp}")
from evaluation.play_attempt import start_play_attempt, finish_play_attempt
_play_status = start_play_attempt(
    checkpoint_path, _play_output, args_cli.seed, args_cli.num_episodes,
    args_cli.video_length, not args_cli.no_video,
)
# Register the directory with error cleanup only after this attempt owns it.
video_dir = _play_output
print(f"[INFO] Output dir : {video_dir}", flush=True)

# ─── 6. Build environment with the same safe USD patching as training ──────────
# After the first stable step a scripted grasp and straight lift run; they are not learned.
env_cfg = UR10eEnvCfg_PLAY()
print(
    "[INFO] Validation-aligned grip trigger: "
    f"xy_tol={GT_SUCCESS_XY_TOL_M:.4f}m, z_tol={SCRIPTED_STAY_Z_TOL_M:.4f}m, "
    f"target_clearance={GT_TARGET_CLEARANCE_ABOVE_CYLINDER_M:.4f}m",
    flush=True,
)
# render_mode="rgb_array" only when video is on; frames are grabbed manually, not via RecordVideo
base_env = make_isaac_env(
    env_cfg, ENV_ID, num_envs=1, seed=args_cli.seed, device=_RUN_DEVICE,
    camera=(args_cli.camera_resolution, args_cli.camera_update_period, "Play"),
    render_mode="rgb_array" if not args_cli.no_video else None,
)
print(f"[GPU] env_cfg.sim.device={env_cfg.sim.device}", flush=True)
PLAY_CAMERA_WIDTH, PLAY_CAMERA_HEIGHT = env_cfg.scene.wrist_camera.width, env_cfg.scene.wrist_camera.height

# ─── 8. SB3 wrapper + frozen VecNormalize + model (shared with the validator) ─
_env_device = str(base_env.unwrapped.device)
_model_device = "cuda" if _env_device.startswith("cuda") else "cpu"
print(f"[INFO] Model type : PPO (checkpoint dir: {checkpoint_tag})", flush=True)
print(f"[INFO] Torch runtime: version={torch.__version__} | file={torch.__file__}", flush=True)
env, model = load_inference_policy(
    base_env, checkpoint_path, vecnorm_path,
    seed=args_cli.seed, device=_model_device, simulation_app=simulation_app,
)
if args_cli.asymmetric_critic and not hasattr(getattr(model, "policy", None), "set_actor_obs_dim"):
    print(
        "[ERROR] Checkpoint metadata says asymmetric_critic=True, but loaded policy has no actor_obs_dim masking.",
        flush=True,
    )
    simulation_app.close()
    sys.exit(1)
print(f"[INFO] Model loaded | device={_model_device} | env_device={_env_device} | obs={model.observation_space.shape}", flush=True)
if hasattr(model.policy, "asymmetric_debug_stats"):
    print(f"[ASYM] play policy stats: {model.policy.asymmetric_debug_stats()}", flush=True)


def _get_scripted_phase0() -> int:
    """Return scripted grip/lift phase for env_0, or 0 before the controller initializes."""
    try:
        isaac_env = base_env.unwrapped
        phase = getattr(isaac_env, "_scripted_grasp_phase", None)
        if phase is None:
            return 0
        return int(phase[0].detach().cpu().item())
    except Exception:
        return 0


def _get_stay_debug_text() -> str:
    """Summarize GT stay condition for scripted grip trigger in env_0."""
    try:
        isaac_env = base_env.unwrapped
        phase = _get_scripted_phase0()
        stay_count_t = getattr(isaac_env, "_scripted_stay_count", None)
        stay_count = int(stay_count_t[0].detach().cpu().item()) if stay_count_t is not None else 0

        robot = isaac_env.scene["robot"]
        body_id = getattr(isaac_env, "_play_gt_debug_ee_body_id", None)
        if body_id is None:
            body_id = find_body_id(robot, env_cfg.actions.arm_action.body_name)
            if body_id is None:
                return f"phase={phase} stay={stay_count}/{SCRIPTED_STAY_STEPS_REQUIRED} gt_body=?"
            setattr(isaac_env, "_play_gt_debug_ee_body_id", body_id)

        # EE − target (reward sign) for env_0.
        err = (-gt_target_error_w(isaac_env, body_id)[0]).detach().float()
        xy_ok = bool((err[0].abs() <= float(GT_SUCCESS_XY_TOL_M)) and (err[1].abs() <= float(GT_SUCCESS_XY_TOL_M)))
        z_ok = bool(err[2].abs() <= float(SCRIPTED_STAY_Z_TOL_M))
        return (
            f"phase={phase} stay={stay_count}/{SCRIPTED_STAY_STEPS_REQUIRED} "
            f"gt_err=({float(err[0]):+.4f},{float(err[1]):+.4f},{float(err[2]):+.4f})m "
            f"xy<={GT_SUCCESS_XY_TOL_M:.4f} {'Y' if xy_ok else 'N'} | "
            f"z<={SCRIPTED_STAY_Z_TOL_M:.4f} {'Y' if z_ok else 'N'}"
        )
    except Exception as e:
        return f"stay_debug=unavailable({e})"


_action_freeze_announced = False

# Fail fast if action-space mismatch (e.g., a checkpoint trained with a different action dim).
try:
    if getattr(env.action_space, "shape", None) != getattr(model.action_space, "shape", None):
        print(f"[ERROR] Action space mismatch: env={env.action_space} vs model={model.action_space}", flush=True)
        simulation_app.close()
        sys.exit(1)
except Exception:
    pass

# ─── 10. Frame capture helper (non-blocking, skipped on failure) ────────────────
import traceback as _traceback

_render_ok = not args_cli.no_video  # starts True, set False on the first failure
if args_cli.no_env_video:
    _render_ok = False

# Wrist camera capture (raw + YOLO-overlay).
_CAM_SENSOR_CFG = SceneEntityCfg("wrist_camera")
camera_frames: list[np.ndarray] = []
camera_det_frames: list[np.ndarray] = []
from evaluation.camera_capture import _to_uint8_rgb, draw_cached_detection

_camera_sensor_warned = False


def _dump_camera_intrinsics() -> None:
    """Persist the REAL wrist-camera intrinsics (fx,fy,cx,cy) to assets/yolo/camera_intrinsics.json.

    This is the authoritative source (IsaacLab TiledCamera.data.intrinsic_matrices) that the YOLO
    dataset generator (extract_dataset_from_usd.py) must match — the standalone generator otherwise
    hardcodes focal/aperture and drifts from the true deploy camera. Written once per play run.
    """
    out_path = os.path.join(ASSETS_ROOT, "yolo", "camera_intrinsics.json")
    try:
        sensor = base_env.unwrapped.scene.sensors.get("wrist_camera")
        if sensor is None:
            return
        K = sensor.data.intrinsic_matrices[0].detach().cpu().numpy()
        payload = {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(PLAY_CAMERA_WIDTH), "height": int(PLAY_CAMERA_HEIGHT),
            "source": "IsaacLab TiledCamera.data.intrinsic_matrices",
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[CAMERA] Real wrist-camera intrinsics dumped → {out_path}: {payload}", flush=True)
    except Exception as e:
        print(f"[CAMERA] Could not dump wrist-camera intrinsics: {e}", flush=True)


def _capture_wrist_camera() -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (raw_rgb, det_rgb) from the env wrist camera; det_rgb draws the policy's YOLO cache."""
    global _camera_sensor_warned
    raw_env = base_env.unwrapped
    try:
        raw_rgb = _to_uint8_rgb(raw_env.scene.sensors["wrist_camera"].data.output["rgb"][0].detach().cpu().numpy())
    except Exception as e:
        if not _camera_sensor_warned:
            print(f"[CAMERA] wrist_camera read failed; camera video skipped: {e}", flush=True)
            _camera_sensor_warned = True
        return None, None
    if raw_rgb is None or raw_rgb.size == 0 or raw_rgb.max() <= 0:
        return None, None
    if args_cli.no_camera_det:
        return raw_rgb, None
    try:
        det_rgb = draw_cached_detection(raw_rgb, _get_yolo_detection_cache(raw_env, _CAM_SENSOR_CFG))
    except Exception as e:
        if not _camera_sensor_warned:
            print(f"[CAMERA] YOLO overlay unavailable: {e}", flush=True)
            _camera_sensor_warned = True
        # Keep the detection video continuous: raw view + exclusion overlay, no box.
        det_rgb = _draw_gripper_exclusion(raw_rgb.copy())
    return raw_rgb, det_rgb


def _capture():
    global _render_ok
    if not _render_ok:
        return None
    try:
        frame = base_env.unwrapped.render()
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            # Skip black frames (usually only the first ones, before the renderer is ready)
            if frame.max() > 0:
                return frame
    except Exception as e:
        print(f"\n[WARNING] render() failed: {e}", flush=True)
        _traceback.print_exc()
        print("[WARNING] Video disabled; play continues.", flush=True)
        _render_ok = False
    return None

# ─── 11. Eval loop ───────────────────────────────────────────────────────────
dt = base_env.unwrapped.step_dt

ep_rewards, ep_lengths, ep_success = [], [], []
_completed_play_episodes = 0
all_frames = []

print(f"\n{'='*60}", flush=True)
print(f"  EVALUASI | {args_cli.num_episodes} ep | max {args_cli.video_length} steps/ep", flush=True)
print(f"  Model    : {checkpoint_tag}", flush=True)
print(f"  Video    : {'SKIP (--no_video)' if args_cli.no_video else 'frame manual (non-blocking)'}", flush=True)
print(f"{'='*60}\n", flush=True)

print("[INFO] Reset env...", flush=True)
obs = env.reset()
print("[INFO] Reset OK.", flush=True)
_dump_camera_intrinsics()
frame = _capture()
if frame is not None:
    all_frames.append(frame)

cam_rgb, cam_det = _capture_wrist_camera()
if cam_rgb is not None:
    camera_frames.append(cam_rgb)
if cam_det is not None:
    camera_det_frames.append(cam_det)

for ep in range(args_cli.num_episodes):
    ep_reward, ep_steps = 0.0, 0
    ep_done = False
    ep_success_flag = False

    print(f"[EP {ep+1}/{args_cli.num_episodes}] start...", flush=True)

    while not ep_done and simulation_app.is_running():
        t0 = time.time()

        with torch.inference_mode():
            actions, _ = model.predict(obs, deterministic=True)

        # Once scripted grip/lift starts, the policy must stop sending arm deltas.
        # The env-side scripted controller holds the stable stay pose during grip
        # and then performs IK lift; letting PPO keep acting here causes camera
        # shake before the gripper has fully closed.
        phase0 = _get_scripted_phase0()
        if phase0 >= 1:
            actions = np.zeros_like(actions)
            if not _action_freeze_announced:
                print("[SCRIPTED] play action freeze active: policy arm deltas disabled", flush=True)
                _action_freeze_announced = True

        obs, rewards, dones, infos = env.step(actions)

        ep_reward += float(rewards[0])
        ep_steps  += 1

        # Capture frame
        frame = _capture()
        if frame is not None:
            all_frames.append(frame)

        cam_rgb, cam_det = _capture_wrist_camera()
        if cam_rgb is not None:
            camera_frames.append(cam_rgb)
        if cam_det is not None:
            camera_det_frames.append(cam_det)

        # In this play cfg the only non-timeout termination is scripted lift success.
        # Sb3VecEnvWrapper exposes termination vs. timeout through this standard flag.
        if dones[0] and infos and isinstance(infos[0], dict):
            ep_success_flag = not bool(infos[0].get("TimeLimit.truncated", False))

        if dones[0] or ep_steps >= args_cli.video_length:
            ep_done = True

        if args_cli.real_time:
            s = dt - (time.time() - t0)
            if s > 0: time.sleep(s)

        # Progress print tiap 30 steps.
        if ep_steps % 30 == 0:
            _dbg = _get_stay_debug_text()
            print(
                f"  step {ep_steps:>3}/{args_cli.video_length} | rew={ep_reward:+.3f} | "
                f"frames={len(all_frames)} | {_dbg}",
                flush=True,
            )

    if ep_done:
        _completed_play_episodes += 1
    ep_rewards.append(ep_reward)
    ep_lengths.append(ep_steps)
    ep_success.append(ep_success_flag)

    status = "SUKSES" if ep_success_flag else "timeout"
    print(f"[EP {ep+1}] {status} | reward={ep_reward:+.3f} | steps={ep_steps}", flush=True)

    if ep < args_cli.num_episodes - 1:
        obs = env.reset()

# ─── 12. Encode video ────────────────────────────────────────────────────────
from evaluation.video import write_mp4

video_path = None
_encoded_videos = []
if all_frames:
    print(f"\n[INFO] Encoding {len(all_frames)} frames → mp4...", flush=True)
    try:
        import imageio
        video_path = os.path.join(video_dir, f"eval_{timestamp}.mp4")
        write_mp4(imageio, video_path, all_frames, fps=30)
        _encoded_videos.append(video_path)
        size_mb = os.path.getsize(video_path) / 1e6
        print(f"[INFO] Video OK: {video_path} ({size_mb:.1f} MB)", flush=True)
    except ImportError:
        print("[WARNING] imageio is missing. Install: pip install imageio[ffmpeg]", flush=True)
    except Exception as e:
        print(f"[WARNING] Encoding failed: {e}", flush=True)
elif not args_cli.no_video:
    print("[INFO] No frames: rendering is unavailable in this setup.", flush=True)

camera_video_path = None
if camera_frames:
    print(f"[INFO] Encoding {len(camera_frames)} wrist camera frames → mp4...", flush=True)
    try:
        import imageio
        camera_video_path = os.path.join(video_dir, f"camera_{timestamp}.mp4")
        write_mp4(imageio, camera_video_path, camera_frames, fps=30)
        _encoded_videos.append(camera_video_path)
        print(f"[INFO] Wrist camera OK: {camera_video_path}", flush=True)
    except Exception as e:
        print(f"[WARNING] Wrist camera encoding failed: {e}", flush=True)
else:
    print("[WARNING] No wrist camera frames (sensor unavailable or not ready).", flush=True)

camera_det_video_path = None
if camera_det_frames:
    print(f"[INFO] Encoding {len(camera_det_frames)} wrist camera det frames → mp4...", flush=True)
    try:
        import imageio
        camera_det_video_path = os.path.join(video_dir, f"camera_det_{timestamp}.mp4")
        write_mp4(imageio, camera_det_video_path, camera_det_frames, fps=30)
        _encoded_videos.append(camera_det_video_path)
        print(f"[INFO] Wrist camera det OK: {camera_det_video_path}", flush=True)
    except Exception as e:
        print(f"[WARNING] Wrist camera detection video encoding failed: {e}", flush=True)
else:
    print("[INFO] camera_det: skip (no detection boxes / no detections).", flush=True)

# ─── 13. Summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}", flush=True)
print("  RINGKASAN", flush=True)
print(f"  Sukses  : {sum(ep_success)}/{args_cli.num_episodes}", flush=True)
print(f"  Reward  : {np.mean(ep_rewards):+.3f} ± {np.std(ep_rewards):.3f}", flush=True)
print(f"  Steps   : {np.mean(ep_lengths):.1f} avg", flush=True)
if video_path: print(f"  Video   : {video_path}", flush=True)
if camera_video_path: print(f"  Camera  : {camera_video_path}", flush=True)
if camera_det_video_path: print(f"  Cam+Det : {camera_det_video_path}", flush=True)
print(f"  CamCfg  : {PLAY_CAMERA_WIDTH}x{PLAY_CAMERA_HEIGHT} | update_period={args_cli.camera_update_period:.4f}s", flush=True)
print(f"{'─'*50}\n", flush=True)

summary_path = os.path.join(video_dir, "eval_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Checkpoint  : {checkpoint_path}\n")
    f.write(f"Timestamp   : {timestamp}\n")
    f.write(f"Seed        : {args_cli.seed}\n")
    f.write(f"Episodes    : {args_cli.num_episodes}\n")
    if video_path: f.write(f"Video       : {video_path}\n")
    if camera_video_path: f.write(f"Wrist camera: {camera_video_path}\n")
    if camera_det_video_path: f.write(f"Cam+Det     : {camera_det_video_path}\n")
    f.write(f"Camera config: {PLAY_CAMERA_WIDTH}x{PLAY_CAMERA_HEIGHT} | update_period={args_cli.camera_update_period:.4f}s\n")
    f.write(f"\n{'─'*40}\n")
    f.write(f"{'EP':>4}  {'Reward':>10}  {'Steps':>6}  {'OK':>4}\n{'─'*40}\n")
    for i, (r, l, s) in enumerate(zip(ep_rewards, ep_lengths, ep_success)):
        f.write(f"{i+1:>4}  {r:>+10.3f}  {l:>6}  {'✔' if s else '✘':>4}\n")
    f.write(f"{'─'*40}\n")
    f.write(f"{'AVG':>4}  {np.mean(ep_rewards):>+10.3f}  {np.mean(ep_lengths):>6.1f}  "
            f"{sum(ep_success)}/{args_cli.num_episodes}\n")
print(f"[INFO] Summary: {summary_path}", flush=True)

# ─── 13b. Training-performance plots ─────────────────────────────────────────
# Render paper-standard figures from the checkpoint's TensorBoard scalars next to
# the eval outputs. checkpoint_dir holds PPO_*/events.out.tfevents.*. Guarded so a
# plotting failure never aborts the eval run.
try:
    from evaluation.training_plots import generate_training_plots
    plots_dir = os.path.join(video_dir, "training_plots")
    generate_training_plots(checkpoint_dir, plots_dir, title=checkpoint_tag)
    print(f"[INFO] Training plots: {plots_dir}", flush=True)
except Exception as e:
    print(f"[PLOTS][WARN] training plot generation skipped: {e}", flush=True)
    _traceback.print_exc()

# ─── 14. Cleanup ─────────────────────────────────────────────────────────────
_play_status = finish_play_attempt(
    video_dir, _play_status, _completed_play_episodes, ep_lengths,
    {"overview": len(all_frames), "camera": len(camera_frames),
     "camera_detection": len(camera_det_frames)}, _encoded_videos,
)
if not _play_status["play_complete"]:
    try:
        env.close()
    finally:
        simulation_app.app.post_uncancellable_quit(2)
        try:
            simulation_app.close(wait_for_replicator=False)
        finally:
            sys.exit(2)
env.close()
print("[INFO] Menutup simulator...", flush=True)
simulation_app.close()
print("[INFO] Selesai.", flush=True)
