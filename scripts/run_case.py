#!/usr/bin/env python3
"""Run a named demo case.

Usage:
    python scripts/run_case.py sanity
    python scripts/run_case.py --list
"""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

CASES = {
    "sanity": "sanity.py",
}


def load_case(name: str):
    """Load a case module by name and return it."""
    filename = CASES[name]
    filepath = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="FEMLab demo runner")
    parser.add_argument("case", nargs="?", help="Name of the case to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    args = parser.parse_args()

    if args.list or args.case is None:
        print("Available cases:")
        for name in sorted(CASES):
            print(f"  {name}")
        sys.exit(0)

    if args.case not in CASES:
        print(f"Unknown case: {args.case}")
        print(f"Available: {', '.join(sorted(CASES))}")
        sys.exit(1)

    mod = load_case(args.case)
    mod.main()


if __name__ == "__main__":
    main()
