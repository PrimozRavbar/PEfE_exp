import csv, glob, sys
for d in sorted(sum([glob.glob(g) for g in sys.argv[1:]], [])):
    try:
        rows = list(csv.DictReader(open(d + "/log.csv")))
    except Exception:
        continue
    if not rows: continue
    v = [(int(r["generation"]), float(r["valid"])) for r in rows if r["valid"]]
    last = rows[-1]
    first12 = next((g for g, x in v if x >= 12.0), None)
    print(f"{d:40s} gen {last['generation']:>4} bestvalid {last['best_valid']:>7} "
          f"lastvalids {[x for _, x in v[-4:]]} meanconn {last['mean_connections']:>7} "
          f"bestconn {last['best_connections']:>5} first>=12 {first12}")
