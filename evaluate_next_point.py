"""
evaluate_next_point.py  -  score a trained KAFY next_point operation.

Put this file in the KAFY repo root (next to kafy_interface.py) and run:
    python evaluate_next_point.py NextPoint data/next_test.csv --resolution 8

It runs the operation on every test trajectory (same code path as RunOperation)
and compares the prediction with the "true_next" column. It also reports a
trivial baseline ("predict the last input point") so you can tell whether the
model learned anything.
"""

import argparse
import math
import os
import statistics
import sys

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)

import h3
import pandas as pd
from commands.runOperation import run_operation


def haversine_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(d))


def parse_point(s):
    lat, lon = s.strip().strip("\"'").split()[:2]
    return float(lat), float(lon)


def summarize(name, errs):
    if not errs:
        print(f"{name:<28} no predictions")
        return
    print(f"{name:<28} mean {statistics.mean(errs):8.0f} m   median {statistics.median(errs):8.0f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("operation_name")
    ap.add_argument("test_csv")
    ap.add_argument("--resolution", type=int, default=8, help="same H3 resolution used for training")
    a = ap.parse_args()

    df = pd.read_csv(a.test_csv)
    df = df[df["trajectory"].notna() & (df["trajectory"].astype(str).str.strip() != "")]
    truths = [parse_point(s) for s in df["true_next"]]
    last_inputs = [parse_point(t.split(",")[-1]) for t in df["trajectory"]]

    results = run_operation(BASE_PATH, a.operation_name, a.test_csv, {"operation": "next_point"})
    if len(results) != len(truths):
        raise SystemExit(f"Got {len(results)} predictions for {len(truths)} test rows; cannot align them.")

    model_errs, base_errs, hits, empty = [], [], 0, 0
    for pred, truth, last in zip(results, truths, last_inputs):
        base_errs.append(haversine_m(last, truth))
        if not pred:
            empty += 1
            continue
        p = pred[0]
        model_errs.append(haversine_m(p, truth))
        if h3.latlng_to_cell(p[0], p[1], a.resolution) == h3.latlng_to_cell(truth[0], truth[1], a.resolution):
            hits += 1

    n = len(truths)
    print("\n=== Next-point evaluation ===")
    print(f"test trajectories:           {n}")
    print(f"exact H3 cell accuracy:      {hits / n:.1%}")
    print(f"no valid cell predicted:     {empty}")
    summarize("model distance error", model_errs)
    summarize("baseline (last point) error", base_errs)


if __name__ == "__main__":
    main()