from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    train_entry = root / "train.py"
    sys.path.insert(0, str(root))
    sys.argv[0] = str(train_entry)
    runpy.run_path(str(train_entry), run_name="__main__")


if __name__ == "__main__":
    main()
