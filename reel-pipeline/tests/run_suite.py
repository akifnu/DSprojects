"""Run core regressions separately from the existing caption-analysis diagnostics.
The default 'all' includes every test and propagates every failure."""
import argparse
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "core", "analysis"), default="all")
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()
    files = sorted(TESTS.glob("test_*.py"))
    if args.suite == "core":
        files = [f for f in files if f.name != "test_sample_text.py"]
    elif args.suite == "analysis":
        files = [TESTS / "test_sample_text.py"]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(f.stem) for f in files)
    result = unittest.TextTestRunner(verbosity=2 if args.v else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
