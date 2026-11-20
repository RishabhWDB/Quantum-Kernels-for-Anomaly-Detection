import argparse
import time

import yaml

from . import report
from .experiments import EXPERIMENTS, Context


def main():
    parser = argparse.ArgumentParser(prog="python -m qkfraud")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run experiments")
    run.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    run.add_argument("--seeds", type=int, nargs="+", help="override the seeds in the config")
    sub.add_parser("report", help="build tables and figures from the result files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.command == "run":
        names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
        ctx = Context(cfg, args.seeds)
        start = time.time()
        for name in names:
            EXPERIMENTS[name](ctx)
        print(f"finished {', '.join(names)} in {(time.time() - start) / 60:.1f} min")
    else:
        print("wrote", report.build(cfg))


if __name__ == "__main__":
    main()
