from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_trl_drgrpo import build_training_dataset


def main() -> int:
    dataset = build_training_dataset("train")
    print(f"train rows: {len(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
