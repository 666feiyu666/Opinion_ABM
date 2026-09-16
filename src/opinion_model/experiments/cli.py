"""Research-facing planning and execution commands."""

import argparse
from pathlib import Path
import tomllib

from opinion_model.experiments.planning import load_design, build_plan, validate_plan
from opinion_model.experiments.runner import export_plan, execute_batch


def main(supporting=False):
    parser = argparse.ArgumentParser(description="Plan or execute descriptive OLIM main/supporting batches.")
    parser.add_argument("--config", type=Path, default=Path("configs/main_experiment/main_grid.toml") if not supporting else None,
                        required=supporting)
    parser.add_argument("--output", type=Path, required=True, help="Descriptive batch folder; never silently overwritten")
    parser.add_argument("--execute", action="store_true", help="Without this flag, write plans only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-batch", type=Path)
    parser.add_argument("--max-runs", type=int, help="Pause after this many newly completed runs")
    parser.add_argument("--reference-only", action="store_true",
                        help="Main only: complete reference runs, then pause the unchanged full-grid batch")
    args = parser.parse_args()
    if args.reference_only and supporting:
        parser.error("--reference-only applies only to the main experiment")
    config, kind = args.config.resolve(), "main"
    if supporting:
        support = tomllib.loads(config.read_text(encoding="utf-8"))["support"]
        kind = support["kind"]
        if kind not in ("reach", "topology", "horizon"):
            parser.error("Unknown support kind")
        config = (config.parent / support["design"]).resolve()
    design = load_design(config)
    plan = build_plan(design)
    template = validate_plan(design, plan)
    if args.execute:
        result = execute_batch(args.output, design, plan, kind, template, resume=args.resume,
                               source_batch=args.source_batch, max_runs=args.max_runs,
                               reference_only=args.reference_only)
        print(f"{kind}: {result['status']} — {args.output.resolve()}")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Plan output directory must be new or empty")
        export_plan(args.output.resolve(), design, plan, kind, template)
        print(f"Plan only: {len(plan.required_ids(kind))} source runs — {args.output.resolve()}")


if __name__ == "__main__":
    main()
