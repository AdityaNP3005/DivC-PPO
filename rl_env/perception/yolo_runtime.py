from __future__ import annotations

"""YOLO runtime: SAM-free Ultralytics import, lazy model singleton, inference
settings, batched box selection, gripper exclusion and box depth."""

import importlib
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


_YOLO_MODEL = None


def _prepare_ultralytics_runtime() -> None:
    """Prepare a YOLO-only Ultralytics runtime for Isaac Sim.

    Torchvision is intentionally cleared: both Isaac's prebundle and the current
    storage venv copy can mismatch the active Torch build. Ultralytics detect
    inference can use its pure TorchNMS path as long as torchvision is not loaded.

    Keep sys.path order untouched. Isaac Sim, SB3, and Torch are sensitive to
    mixing packages from Isaac's Python tree and a separate package tree after startup.
    """
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)


def _patch_ultralytics_network_checks() -> None:
    """Disable Ultralytics GitHub asset checks for offline/restricted environments."""
    try:
        from ultralytics.utils import downloads as ultralytics_downloads
        try:
            from ultralytics.utils import SETTINGS as ultralytics_settings
        except Exception:
            ultralytics_settings = None

        ultralytics_downloads.get_github_assets = lambda *args, **kwargs: ("", [])
        ultralytics_downloads.attempt_download_asset = lambda file, *args, **kwargs: str(file)
        if ultralytics_settings is not None:
            try:
                ultralytics_settings.update({"sync": False})
            except Exception:
                pass
    except Exception:
        # Do not interrupt training/validation if the patch is unavailable.
        pass


def _import_yolo_class_without_sam():
    """Import Ultralytics YOLO without importing SAM/torchvision.

    Ultralytics 8.4 lazy `from ultralytics import YOLO` imports
    `ultralytics.models.__init__`, which imports SAM and then torchvision.
    Isaac Sim often ships a mismatched torchvision, so for this detection-only
    pipeline we load only `ultralytics.models.yolo.model.YOLO`.
    """
    _prepare_ultralytics_runtime()
    import ultralytics

    models_dir = Path(ultralytics.__file__).resolve().parent / "models"
    for name in list(sys.modules):
        if name == "ultralytics.models" or name.startswith("ultralytics.models."):
            sys.modules.pop(name, None)

    models_pkg = types.ModuleType("ultralytics.models")
    models_pkg.__path__ = [str(models_dir)]
    models_pkg.__package__ = "ultralytics.models"
    models_pkg.__file__ = str(models_dir / "__init__.py")
    sys.modules["ultralytics.models"] = models_pkg
    setattr(ultralytics, "models", models_pkg)

    return importlib.import_module("ultralytics.models.yolo.model").YOLO


def _get_yolo_model():
    global _YOLO_MODEL
    _prepare_ultralytics_runtime()
    if _YOLO_MODEL is None:
        _patch_ultralytics_network_checks()
        YOLO = _import_yolo_class_without_sam()

        from config import YOLO_DEFAULTS
        from evaluation.checkpoint_assets import require_detector

        # Entrypoints publish YOLO_MODEL_PATH (per-object detector); a missing file is a hard
        # error, never a silent fallback to a generic detector.
        model_path = require_detector(os.environ.get("YOLO_MODEL_PATH") or YOLO_DEFAULTS["model_path"])
        print(f"[YOLO] loading weights from: {model_path}")
        try:
            _YOLO_MODEL = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}") from e
    return _YOLO_MODEL


def _get_yolo_settings() -> tuple[float, float, str, int]:
    conf = float(os.environ.get("YOLO_CONF", "0.25"))
    iou = float(os.environ.get("YOLO_IOU", "0.45"))
    device = os.environ.get("YOLO_DEVICE", "cpu")
    max_det = int(os.environ.get("YOLO_MAX_DET", "1"))
    return conf, iou, device, max_det


@lru_cache(maxsize=1)
def _tensor_box_predictor():
    """Ultralytics detection predictor for tensor batches that are the original frames (no letterbox): boxes only.

    The stock postprocess copies every input image back to a numpy array just to store Results.orig_img and runs NMS
    in a Python loop per image (about 87 ms per step at 128 envs). Here the kept boxes of the whole batch are left
    on predictor.top (see _top_boxes) in the frame's pixel coordinates (clipped to it, as stock). List (numpy)
    inputs keep the stock path.
    """
    _prepare_ultralytics_runtime()
    _import_yolo_class_without_sam()
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    class TensorBoxPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            if isinstance(orig_imgs, list):
                return super().postprocess(preds, img, orig_imgs, **kwargs)
            if self.args.classes is not None or self.args.agnostic_nms or getattr(self.model, "end2end", False):
                raise RuntimeError("TensorBoxPredictor covers plain per-class detection NMS only")
            self.top = _top_boxes(preds, self.args.conf, self.args.iou, self.args.max_det, img.shape[2:])
            return [types.SimpleNamespace() for _ in orig_imgs]  # predict() only sets .speed on these

    return TensorBoxPredictor


def _scaled_gripper_exclusion_polygons_px(
    image_width: int = 640,
    image_height: int = 640,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    from config import (
        YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX,
        YOLO_GRIPPER_EXCLUSION_POLYGONS_PX,
    )

    canonical_w, canonical_h = YOLO_GRIPPER_EXCLUSION_CANONICAL_SIZE_PX
    scale_x = float(image_width) / float(canonical_w)
    scale_y = float(image_height) / float(canonical_h)
    return tuple(
        tuple((float(x) * scale_x, float(y) * scale_y) for x, y in polygon)
        for polygon in YOLO_GRIPPER_EXCLUSION_POLYGONS_PX
    )


def _centres_in_gripper_exclusion(cx: torch.Tensor, cy: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """True where a box centre lies inside or on the edge of a gripper polygon (all False when YOLO_ROI_ENABLED=0)."""
    from config import YOLO_ROI_ENABLED

    inside_any = torch.zeros_like(cx, dtype=torch.bool)
    if not YOLO_ROI_ENABLED:
        return inside_any
    px, py = cx.double(), cy.double()
    for polygon in _scaled_gripper_exclusion_polygons_px(image_width, image_height):
        inside, on_edge = torch.zeros_like(inside_any), torch.zeros_like(inside_any)
        for index, (x1, y1) in enumerate(polygon):  # even-odd ray casting, edges count as inside
            x2, y2 = polygon[(index + 1) % len(polygon)]
            cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
            on_edge |= ((cross.abs() <= 1e-7) & (px >= min(x1, x2) - 1e-7) & (px <= max(x1, x2) + 1e-7)
                        & (py >= min(y1, y2) - 1e-7) & (py <= max(y1, y2) + 1e-7))
            if y1 != y2:
                inside ^= ((y1 > py) != (y2 > py)) & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
        inside_any |= inside | on_edge
    return inside_any


def _top_boxes(preds, conf_thres: float, iou_thres: float, max_det: int, shape, max_wh: int = 7680):
    """Ultralytics non_max_suppression + clip_boxes for the whole batch at once.

    (boxes [B, max_det, 6] = xyxy, conf, class; kept [B, max_det]), highest confidence first. The greedy rule is
    TorchNMS.nms (keep the best box, drop the rest above the IoU threshold, repeat), run on all images together.
    """
    pred = (preds[0] if isinstance(preds, (list, tuple)) else preds).transpose(1, 2)  # [B, N, 4 + nc]
    conf, cls = pred[..., 4:].max(-1)
    cand = conf > conf_thres
    k = max(int(cand.sum(1).max()), 1)
    conf, order = torch.where(cand, conf, -1.0).topk(k, dim=1)  # sorted, candidates first
    alive = conf > conf_thres
    cls = cls.gather(1, order).float()
    xywh = pred[..., :4].gather(1, order.unsqueeze(-1).expand(-1, -1, 4))
    xyxy = torch.cat([xywh[..., :2] - xywh[..., 2:] / 2, xywh[..., :2] + xywh[..., 2:] / 2], -1)
    x1, y1, x2, y2 = (xyxy + cls.unsqueeze(-1) * max_wh).unbind(-1)  # classes never suppress each other
    area = (x2 - x1) * (y2 - y1)
    rows = torch.arange(pred.shape[0], device=pred.device)
    picks, kept = [], []
    for _ in range(max_det):
        i = (alive.cumsum(1) == 0).sum(1)  # first box still alive
        kept.append(i < k)
        i = i.clamp(max=k - 1)
        picks.append(i)
        alive[rows, i] = False
        pick = lambda v: v[rows, i].unsqueeze(1)
        inter = ((torch.minimum(pick(x2), x2) - torch.maximum(pick(x1), x1)).clamp_(min=0)
                 * (torch.minimum(pick(y2), y2) - torch.maximum(pick(y1), y1)).clamp_(min=0))
        no_overlap = (inter * alive).sum(1, keepdim=True) == 0  # stock early exit: nothing touches the kept box
        alive &= no_overlap | (inter / (pick(area) + area - inter) <= iou_thres)
    picks, kept = torch.stack(picks, 1), torch.stack(kept, 1)
    boxes = torch.cat([xyxy, conf.unsqueeze(-1), cls.unsqueeze(-1)], -1).gather(1, picks.unsqueeze(-1).expand(-1, -1, 6))
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])
    return boxes, kept


def _best_unmasked_detections(boxes: torch.Tensor, kept: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per image: the most confident kept box whose centre is outside the gripper polygons, with its depths (all on the GPU).

    boxes/kept from _top_boxes, depth [B, H, W] in metres. Returns xyxy [B, 4], conf, has_det, depth_area (the
    YOLO_DEPTH_PERCENTILE of the valid, unmasked depth pixels in the box, 0 when none) and depth (depth_area, else the
    centre pixel, else 0). Images without a detection get zeros.
    """
    from config import YOLO_DEPTH_PERCENTILE

    boxes, kept = boxes.to(depth.device), kept.to(depth.device)  # the detector may sit on another device than the sim
    n, (h, w) = boxes.shape[0], depth.shape[1:]
    rows = torch.arange(n, device=depth.device)
    mask = torch.as_tensor(_gripper_exclusion_mask(h, w), device=depth.device)
    px = lambda v, top: v.long().clamp(0, top - 1)
    centre = (boxes[..., :2] + boxes[..., 2:4]) * 0.5
    ok = kept & ~_centres_in_gripper_exclusion(centre[..., 0], centre[..., 1], w, h)
    first = (ok.cumsum(1) == 0).sum(1)  # boxes are sorted by confidence
    has = first < boxes.shape[1]
    box = boxes[rows, first.clamp(max=boxes.shape[1] - 1)] * has.unsqueeze(1)
    x1, y1, x2, y2 = px(box[:, 0], w), px(box[:, 1], h), px(box[:, 2], w), px(box[:, 3], h)
    in_box = has & (x1 < x2) & (y1 < y2)
    # one padded crop per image, as large as the largest box of this batch
    ys = y1[:, None] + torch.arange(int((y2 - y1).max()) + 1, device=depth.device)
    xs = x1[:, None] + torch.arange(int((x2 - x1).max()) + 1, device=depth.device)
    yy, xx = ys.clamp(max=h - 1)[:, :, None], xs.clamp(max=w - 1)[:, None, :]
    crop = depth[rows[:, None, None], yy, xx].float()
    good = ((ys <= y2[:, None])[:, :, None] & (xs <= x2[:, None])[:, None, :] & in_box[:, None, None]
            & torch.isfinite(crop) & (crop > 0.0) & ~mask[yy, xx]).flatten(1)
    values = torch.where(good, crop.flatten(1), torch.inf).sort(1).values
    count = good.sum(1)
    rank = torch.tensor(float(YOLO_DEPTH_PERCENTILE), device=depth.device) * (count - 1).clamp(min=0)  # torch.quantile
    below, above = values[rows, rank.floor().long()], values[rows, rank.ceil().long()]
    area = torch.lerp(below, above, rank - rank.floor())
    area = torch.where((count > 0) & torch.isfinite(area), area, 0.0)
    cx, cy = ((box[:, 0] + box[:, 2]) * 0.5).round(), ((box[:, 1] + box[:, 3]) * 0.5).round()
    cx, cy = px(cx, w), px(cy, h)
    point = depth[rows, cy, cx].float()
    point = torch.where(has & ~mask[cy, cx] & torch.isfinite(point), point, 0.0)
    return {"xyxy": box[:, :4], "conf": box[:, 4], "has_det": has.float(),
            "depth_area": area, "depth": torch.where(area > 0.0, area, point)}


@lru_cache(maxsize=8)
def _gripper_exclusion_mask(image_height: int, image_width: int) -> np.ndarray:
    from config import YOLO_ROI_ENABLED

    mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    if not YOLO_ROI_ENABLED:
        return mask.astype(bool)
    try:
        import cv2
    except Exception:
        return mask.astype(bool)
    polygons = [
        np.asarray(polygon, dtype=np.int32)
        for polygon in _scaled_gripper_exclusion_polygons_px(image_width, image_height)
    ]
    cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)


def _draw_gripper_exclusion(img, color=(255, 0, 0), alpha: float = 0.28, thickness: int = 2):
    from config import YOLO_ROI_ENABLED

    if not YOLO_ROI_ENABLED or img is None:
        return img
    try:
        import cv2
    except Exception:
        return img
    height, width = img.shape[:2]
    polygons = [
        np.asarray(polygon, dtype=np.int32)
        for polygon in _scaled_gripper_exclusion_polygons_px(width, height)
    ]
    overlay = img.copy()
    cv2.fillPoly(overlay, polygons, color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    cv2.polylines(img, polygons, True, color, thickness)
    cv2.putText(
        img,
        "GRIPPER EXCLUSION",
        (10, max(24, height - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return img
