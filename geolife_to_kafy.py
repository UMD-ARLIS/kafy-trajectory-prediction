"""
geolife_to_kafy.py
Convert GeoLife Trajectories 1.3 (.plt files) into CSVs for KAFY's next_point operation.

Usage:
    python geolife_to_kafy.py "/path/to/Geolife Trajectories 1.3/Data" --out-dir data --resolution 8

Produces in --out-dir:
    next_train.csv   column "trajectory": "lat lon,lat lon,..."   (KAFY splits off the last point itself)
    next_test.csv    column "trajectory": input points (last point removed)
                     column "true_next":  "lat lon" of the removed point (ground truth)

What it does to the raw GPS:
  * keeps only points inside a bounding box (default: greater Beijing), so every
    trajectory falls in the same KAFY partitioning cell
  * splits a .plt file wherever there is a time gap > --max-gap-min minutes
  * collapses consecutive points that fall in the same H3 cell (GeoLife logs every
    1-5 s, so without this the "next point" is almost always the same cell)
  * cuts the resulting cell sequences into non-overlapping windows of --window points
  * splits train/test by USER, so the test set has no trips from training users

--resolution MUST match the resolution you pass to KAFY in args[resolution=...].
"""

import argparse
import csv
import os
import random
from datetime import datetime

import h3

BEIJING_BBOX = (39.6, 40.4, 115.9, 117.0)  # min_lat, max_lat, min_lon, max_lon


def read_plt(path):
    """Yield (lat, lon, datetime) from a GeoLife .plt file (6 header lines)."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i < 6:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat, lon = float(parts[0]), float(parts[1])
                t = datetime.strptime(parts[5] + " " + parts[6], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            yield lat, lon, t


def segments_from_plt(path, bbox, max_gap_s, resolution):
    """Return a list of segments; each is a list of (lat, lon) with no repeated consecutive H3 cell."""
    min_lat, max_lat, min_lon, max_lon = bbox
    segments, current = [], []
    last_cell, last_t = None, None

    def close():
        nonlocal current, last_cell
        if current:
            segments.append(current)
        current, last_cell = [], None

    for lat, lon, t in read_plt(path):
        inside = min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        gap = last_t is not None and (t - last_t).total_seconds() > max_gap_s
        last_t = t
        if not inside or gap:
            close()
            if not inside:
                continue
        cell = h3.latlng_to_cell(lat, lon, resolution)
        if cell != last_cell:
            current.append((lat, lon))
            last_cell = cell
    close()
    return segments


def windows(seq, size, min_len):
    for i in range(0, len(seq), size):
        w = seq[i:i + size]
        if len(w) >= min_len:
            yield w


def fmt(points):
    return ",".join(f"{lat:.6f} {lon:.6f}" for lat, lon in points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="GeoLife 'Data' folder (contains 000, 001, ...)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--resolution", type=int, default=8, help="H3 resolution (match KAFY args)")
    ap.add_argument("--window", type=int, default=20, help="points per trajectory window")
    ap.add_argument("--min-len", type=int, default=6, help="drop windows shorter than this")
    ap.add_argument("--max-gap-min", type=float, default=10, help="split trips at gaps longer than this")
    ap.add_argument("--bbox", type=float, nargs=4, default=BEIJING_BBOX,
                    metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"))
    ap.add_argument("--test-every", type=int, default=10,
                    help="users whose numeric id %% N == 0 go to the test set")
    ap.add_argument("--max-train", type=int, default=None, help="randomly keep at most N training rows")
    ap.add_argument("--max-test", type=int, default=None, help="randomly keep at most N test rows")
    ap.add_argument("--max-users", type=int, default=None, help="only read the first N users (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    users = sorted(d for d in os.listdir(a.data_dir) if os.path.isdir(os.path.join(a.data_dir, d, "Trajectory")))
    if a.max_users:
        users = users[:a.max_users]
    if not users:
        raise SystemExit(f"No user folders with a Trajectory/ subfolder found in {a.data_dir}")

    train, test = [], []
    for u in users:
        is_test = int(u) % a.test_every == 0
        tdir = os.path.join(a.data_dir, u, "Trajectory")
        for name in sorted(os.listdir(tdir)):
            if not name.endswith(".plt"):
                continue
            for seg in segments_from_plt(os.path.join(tdir, name), a.bbox, a.max_gap_min * 60, a.resolution):
                for w in windows(seg, a.window, a.min_len):
                    (test if is_test else train).append(w)
        print(f"user {u}: train={len(train)} test={len(test)}", end="\r")
    print()

    rng = random.Random(a.seed)
    if a.max_train and len(train) > a.max_train:
        train = rng.sample(train, a.max_train)
    if a.max_test and len(test) > a.max_test:
        test = rng.sample(test, a.max_test)

    os.makedirs(a.out_dir, exist_ok=True)
    train_path = os.path.join(a.out_dir, "next_train.csv")
    test_path = os.path.join(a.out_dir, "next_test.csv")

    with open(train_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["trajectory"])
        for w in train:
            wr.writerow([fmt(w)])

    with open(test_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["trajectory", "true_next"])
        for w in test:
            wr.writerow([fmt(w[:-1]), fmt(w[-1:])])

    print(f"Wrote {len(train)} training trajectories -> {train_path}")
    print(f"Wrote {len(test)} test trajectories     -> {test_path}")
    print(f"Remember: use args[resolution={a.resolution},...] in AddOperation.")


if __name__ == "__main__":
    main()