"""Plan or execute reach, topology, or horizon supporting experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opinion_model.experiments.cli import main

if __name__ == "__main__":
    main(supporting=True)
