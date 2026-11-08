"""The evolve.py defaults must be exactly the "B apc" setup.

Three checks:
  1. the parsed defaults equal the explicit B apc flags (fast);
  2. a 40-generation seed-0 run with no flags reproduces, row for row, the
     log.csv recorded *before* the defaults changed (run then with explicit
     flags; tests/data/apc_seed0_gen40_log.csv).  The wall-clock `seconds`
     column is ignored; every other column must be identical;
  3. evaluate.py on that run's best.npz gives 15.62 +- 0.83, blind 0.23,
     82 connections (best validation 15.82).

Run: python tests/test_defaults.py        (about 1 minute)
 or: pytest tests/test_defaults.py
"""
import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import evolve  # noqa: E402
import evaluate  # noqa: E402

APC_FLAGS = ["--init", "zeros", "--add-per-child", "3", "--add-share", "0.4,0.2,0.4",
             "--remove-rate", "0.02", "--connection-scale", "10", "--generations", "150"]
GOLDEN_LOG = os.path.join(HERE, "data", "apc_seed0_gen40_log.csv")


def _rows(path):
    with open(path) as f:
        return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(f)]


def test_parsed_defaults_are_apc():
    p = evolve.build_parser()
    d = vars(p.parse_args(["--out", "x"]))
    e = vars(p.parse_args(["--out", "x"] + APC_FLAGS))
    assert d == e, {k: (d[k], e[k]) for k in d if d[k] != e[k]}
    assert d["generations"] == 150 and d["add_per_child"] == 3.0


def test_default_run_matches_golden():
    with tempfile.TemporaryDirectory() as out:
        evolve.main(["--out", out, "--seed", "0", "--generations", "40"])
        assert _rows(os.path.join(out, "log.csv")) == _rows(GOLDEN_LOG)
        r = evaluate.run_genome(os.path.join(out, "best.npz"))
    assert r["meta"]["valid"] == 15.82
    assert round(r["score"]["mean"], 2) == 15.62
    assert round(r["score"]["sd"], 2) == 0.83
    assert round(r["blind"]["mean"], 2) == 0.23
    assert r["connections"] == 82


if __name__ == "__main__":
    test_parsed_defaults_are_apc()
    print("OK: parsed defaults == B apc flags")
    test_default_run_matches_golden()
    print("OK: default 40-gen run matches golden log.csv and evaluation")
