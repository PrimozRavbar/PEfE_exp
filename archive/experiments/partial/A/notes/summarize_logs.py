"""Print validation trajectory of evolve.py runs: python summarize_logs.py RUN_DIR..."""
import csv
import sys

for d in sys.argv[1:]:
    rows = list(csv.DictReader(open(f"{d}/log.csv")))
    v = {int(r["generation"]): r["valid"] for r in rows if r["valid"]}
    pick = [g for g in (19, 39, 69, 99, 149) if g in v]
    last = rows[-1]
    print(f"{d:40s} " + " ".join(f"v{g+1}={float(v[g]):5.2f}" for g in pick)
          + f" best_valid={float(last['best_valid']):5.2f} mean={float(last['mean']):5.2f}"
          f" conns={last['best_connections']}/{float(last['mean_connections']):.0f}"
          f" s/gen={sum(float(r['seconds']) for r in rows)/len(rows):.2f}")
