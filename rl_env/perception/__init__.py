"""Wrist-camera perception: YOLO runtime (`yolo_runtime`), the per-step detection
cache (`yolo_cache`, the only YOLO inference path) and observation terms
(`obs_functions`). Import the submodules directly; this package imports nothing so
standalone helpers can load `yolo_runtime` after SimulationApp without Isaac Lab.
"""
