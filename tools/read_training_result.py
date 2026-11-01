"""Resolve this chain's exact completed model, never scan another run's latest."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.validation_contract import file_digest


def resolve(path):
    result = json.loads(Path(path).read_text())
    if not result.get("training_complete"):
        raise ValueError("Training did not complete its requested budget; post-training evaluation refused.")
    for key, digest in (("checkpoint", "checkpoint_sha256"), ("normalizer", "normalizer_sha256")):
        if file_digest(result[key]) != result[digest]:
            raise ValueError(f"Training result file changed: {result[key]}")
    return result["checkpoint"]


if __name__ == "__main__":
    print(resolve(sys.argv[1]))
