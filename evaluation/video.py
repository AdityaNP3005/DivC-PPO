"""MP4 encoding primitive; callers own optional imports, paths and error policy."""
import numpy as np


def write_mp4(writer, path, frames, fps=30):
    """Encode uint8 copies using the caller's lazily imported imageio module."""
    writer.mimsave(path, [frame.astype(np.uint8) for frame in frames], fps=fps)
