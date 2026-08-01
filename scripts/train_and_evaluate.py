import subprocess
import sys
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/build_balanced_dataset.py"], cwd=base, check=True)
    subprocess.run([sys.executable, "scripts/retrain.py"], cwd=base, check=True)
    subprocess.run([sys.executable, "evaluate.py"], cwd=base, check=True)


if __name__ == "__main__":
    main()
