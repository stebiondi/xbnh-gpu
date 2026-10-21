"""
Compare point output of Estero runs (XBeach fortran point files: per record t, zs, uu, vv, u, v as float64).

For each run directory against the reference run, over t >= tspin and for every output point:
  zs correlation, zs RMS difference, Hs = 4 std(detrended zs), mean u and v, max |(u, v)|.
Reported per run: medians over the points of the relative differences in Hs, mean current and max speed,
and the zs correlation. Extra CPU runs (e.g. the friction sensitivity tests) give the physical signal scale.
Usage: python tools/compare_points.py <ref dir> <run dir> [<run dir> ...] [--tspin 30] [--json f]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def points(d):
    lines = (d / "params.txt").read_text().splitlines()
    for n, line in enumerate(lines):
        if line.split("=")[0].strip().lower() == "npoints":
            return [l.split()[2] for l in lines[n + 1:n + 1 + int(line.split("=")[1])]]
    return []


def stats(a, tspin):
    t = a[:, 0]
    s = t >= tspin
    z = a[s, 1]
    zd = z - np.polyval(np.polyfit(t[s], z, 1), t[s])
    u, v = a[s, 4], a[s, 5]
    return dict(Hs=4 * zd.std(), um=u.mean(), vm=v.mean(), smax=np.hypot(u, v).max(), z=z, t=t[s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tspin", type=float, default=30.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    R = Path(a.ref)
    names = points(R)
    ref = {n: stats(np.fromfile(R / f"{n}.dat", "<f8").reshape(-1, 6), a.tspin) for n in names}
    out = {"ref": str(R), "npoints": len(names), "tspin": a.tspin, "runs": {}}
    for run in a.runs:
        P = Path(run)
        rows = {}
        for n in names:
            b = np.fromfile(P / f"{n}.dat", "<f8").reshape(-1, 6)
            s, r = stats(b, a.tspin), ref[n]
            k = min(len(r["z"]), len(s["z"]))
            cur_r = np.hypot(r["um"], r["vm"])
            rows[n] = dict(zs_corr=float(np.corrcoef(r["z"][:k], s["z"][:k])[0, 1]),
                           zs_rms_diff=float(np.sqrt(np.mean((r["z"][:k] - s["z"][:k]) ** 2))),
                           dHs_rel=float((s["Hs"] - r["Hs"]) / r["Hs"]),
                           dmean_current=float(np.hypot(s["um"] - r["um"], s["vm"] - r["vm"])),
                           mean_current_ref=float(cur_r),
                           dsmax_rel=float((s["smax"] - r["smax"]) / r["smax"]))
        med = lambda key: float(np.median([abs(v[key]) for v in rows.values()]))
        mean_cur = float(np.median([v["mean_current_ref"] for v in rows.values()]))
        out["runs"][run] = dict(median_zs_corr=float(np.median([v["zs_corr"] for v in rows.values()])),
                                median_zs_rms_diff_m=med("zs_rms_diff"),
                                median_abs_dHs_rel=med("dHs_rel"),
                                median_dmean_current_ms=med("dmean_current"),
                                median_mean_current_ref_ms=mean_cur,
                                median_abs_dsmax_rel=med("dsmax_rel"),
                                points=rows)
        o = out["runs"][run]
        print(f"{run}\n  zs corr {o['median_zs_corr']:.3f}  zs rms diff {o['median_zs_rms_diff_m']:.3f} m  "
              f"|dHs| {100 * o['median_abs_dHs_rel']:.1f} %  |d mean current| {o['median_dmean_current_ms']:.3f} m/s "
              f"(ref {mean_cur:.3f})  |d max speed| {100 * o['median_abs_dsmax_rel']:.1f} %")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
