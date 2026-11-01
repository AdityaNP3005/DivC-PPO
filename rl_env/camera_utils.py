"""Camera data primitives; no Isaac, detector, OpenCV or plotting imports.

Training and capture keep separate RGB contracts: capture validates shapes and
casts float32; training retains its original dtype and clipping order.
"""
import numpy as np


def parse_resolution(res_text, default=(640, 640)):
    try:
        w_str, h_str = (res_text or '').lower().split('x', 1)
        w, h = int(w_str), int(h_str)
        return max(w, 16), max(h, 16)
    except Exception:
        return default


def configure_camera_sensor(env_cfg, resolution_text, update_period, label):
    """Apply the configured wrist-camera resolution/period (training and validation).

    Returns (width, height). Warns when the size differs from the YOLO calibration.
    """
    from config import CAMERA_HEIGHT, CAMERA_WIDTH

    width, height = parse_resolution(resolution_text, default=(CAMERA_WIDTH, CAMERA_HEIGHT))
    sensor = getattr(getattr(env_cfg, 'scene', None), 'wrist_camera', None)
    if sensor is None:
        return width, height
    sensor.width = int(width)
    sensor.height = int(height)
    sensor.update_period = max(float(update_period), 0.0)
    print(
        f"[CAMERA] {label} sensor: {type(sensor).__name__} @ {sensor.width}x{sensor.height}, "
        f"update_period={sensor.update_period:.4f}s",
        flush=True,
    )
    if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
        print(
            "[WARNING] camera_resolution differs from YOLO calibration "
            f"({CAMERA_WIDTH}x{CAMERA_HEIGHT}); check camera calibration if visual quality shifts.",
            flush=True,
        )
    return width, height


def training_rgb_uint8(img):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    maxv = float(np.max(arr)) if arr.size > 0 else 0.0
    if maxv <= 1.0:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def capture_rgb_uint8(img):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr
