"""Deterministic checkpoint validation."""
from evaluation.inference_setup import setup
from evaluation.inference_run import run
from evaluation.runtime_cleanup import close_failed_app


def main():
    try:
        run(setup())
    except BaseException as error:
        close_failed_app(error)
        raise


if __name__ == "__main__":
    main()
