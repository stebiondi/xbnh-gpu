"""
Compare xbnh_gpu against Fortran XBeach on the same deck (global zs, u, v, float64 fortran output).

Metrics (statistics over t >= tspin):
  - zs time series at gauges: correlation and RMS difference relative to the reference std,
  - significant wave height Hs = 4 std(zs) per cell: map difference over wet cells,
  - mean current |<u>,<v>| per cell (rip and longshore circulation),
  - mean set-up <zs>.
Usage: python tools/compare_rip.py <ref dir> <test dir> [--tspin 60] [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_params(d):
    p = {}
    for line in (d / "params.txt").read_text().splitlines():
        line = line.split("%")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            p[k.strip().lower()] = v.strip()
    return p


def load(d, var, n):
    a = np.fromfile(d / f"{var}.dat", "<f8")
    return a.reshape(-1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tspin", type=float, default=60.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    R, T = Path(a.ref), Path(a.test)
    p = read_params(R)
    nx, ny, dx = int(p["nx"]), int(p["ny"]), float(p["dx"])
    tintg = float(p["tintg"])
    n = (nx + 1) * (ny + 1)
    zb = np.loadtxt(R / "z.grd")
    out = {}
    f = {}
    for v in ("zs", "u", "v"):
        r, t = load(R, v, n), load(T, v, n)
        m = min(len(r), len(t))
        f[v] = (r[:m].reshape(m, ny + 1, nx + 1), t[:m].reshape(m, ny + 1, nx + 1))
    m = f["zs"][0].shape[0]
    time = np.arange(m) * tintg
    sel = time >= a.tspin
    zr, zt = f["zs"]
    wet = ((zr[sel] - zb) > 0.05).all(axis=0)
    out["frames"] = int(m)
    # gauges (x [m], y [m]) : offshore, bar crest outside rip, rip channel, inner surf, inner surf behind rip
    yc = 0.5 * ny * dx
    gauges = {"offshore_x100": (100, yc + 80), "bar_x300": (300, yc + 80), "rip_x300": (300, yc),
              "trough_x360": (360, yc + 80), "inner_x420": (420, yc)}
    out["gauges"] = {}
    for name, (x, y) in gauges.items():
        i, j = int(round(x / dx)), int(round(y / dx))
        a1, b1 = zr[sel, j, i], zt[sel, j, i]
        c = np.corrcoef(a1, b1)[0, 1]
        e = np.sqrt(np.mean((a1 - b1) ** 2)) / np.std(a1)
        hs_r, hs_t = 4 * np.std(a1), 4 * np.std(b1)
        out["gauges"][name] = dict(corr=round(float(c), 4), rms_rel=round(float(e), 4),
                                   Hs_ref=round(float(hs_r), 3), Hs_test=round(float(hs_t), 3))
    hs_r, hs_t = 4 * zr[sel].std(axis=0), 4 * zt[sel].std(axis=0)
    dh = (hs_t - hs_r)[wet]
    out["Hs_map"] = dict(mean_ref=round(float(hs_r[wet].mean()), 4), bias=round(float(dh.mean()), 4),
                         rmse=round(float(np.sqrt((dh ** 2).mean())), 4),
                         rel_rmse=round(float(np.sqrt((dh ** 2).mean()) / hs_r[wet].mean()), 4))
    setr, sett = zr[sel].mean(axis=0), zt[sel].mean(axis=0)
    ds = (sett - setr)[wet]
    out["setup_map"] = dict(rmse_m=round(float(np.sqrt((ds ** 2).mean())), 5))
    ur, ut = f["u"][0][sel].mean(axis=0), f["u"][1][sel].mean(axis=0)
    vr, vt = f["v"][0][sel].mean(axis=0), f["v"][1][sel].mean(axis=0)
    sr, st = np.hypot(ur, vr), np.hypot(ut, vt)
    du = np.hypot(ut - ur, vt - vr)[wet]
    out["mean_current"] = dict(max_ref=round(float(sr[wet].max()), 4), max_test=round(float(st[wet].max()), 4),
                               rmse_vec=round(float(np.sqrt((du ** 2).mean())), 4),
                               rel_rmse=round(float(np.sqrt((du ** 2).mean()) / np.sqrt((sr[wet] ** 2).mean())), 4))
    # instantaneous field difference, all frames after spin-up
    diff = zt[sel] - zr[sel]
    out["zs_field"] = dict(rms_diff=round(float(np.sqrt(np.mean(diff[:, wet] ** 2))), 5),
                           rms_ref=round(float(np.std(zr[sel][:, wet])), 5))
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
