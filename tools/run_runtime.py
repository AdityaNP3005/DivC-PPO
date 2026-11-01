"""Run a project entrypoint with Isaac error cleanup, including bootstrap errors."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_runtime.py <project-script.py> [arguments]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():
        raise ValueError("Runtime entrypoint must be a file inside this project")
    sys.argv = [str(script), *sys.argv[2:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            from evaluation.runtime_cleanup import close_failed_app
            close_failed_app(error)
        raise


if __name__ == "__main__":
    main()
