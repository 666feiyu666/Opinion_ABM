"""Plot one matched setting from any completed main or supporting batch."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opinion_model.visualization.results import load_figure_data, render_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--population", type=int)
    parser.add_argument("--topology")
    parser.add_argument("--leader-share", type=float)
    parser.add_argument("--reach", type=float)
    parser.add_argument("--through-round", type=int)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    data = load_figure_data(args.batch, population=args.population, topology=args.topology,
                            leader_share=args.leader_share, reach=args.reach, through_round=args.through_round)
    print(render_figures(data, args.output, tuple(args.formats), args.dpi))


if __name__ == "__main__":
    main()
