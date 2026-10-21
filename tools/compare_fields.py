"""
Compare two runs of a deck with global zs, u, v output (e.g. examples/estero_features), e.g. xbnh_gpu against
Fortran XBeach MPI, or XBeach MPI-2 against MPI-4 as the decomposition yardstick.

Metrics over t >= tspin (global zs, u, v every tintg; point files every tintp):
  - Hs = 4 std(zs) per wet cell after removing the linear tide trend: relative RMSE of the map,
  - mean current vector per wet cell: relative RMSE; alongshore-mean longshore current profile V(x),
  - mean water level (set-up + tide) RMSE, instantaneous zs field RMS difference,
  - point time series (zs, u, v): correlation and RMS difference,
  - periodic seam: max |zs(row 1) - zs(row ny-2)| (zero for an exact periodic halo).
Usage: python tools/compare_fields.py <ref dir> <test dir> [--tspin 60] [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_params(d):
    p = {}
    lines = (d / "params.txt").read_text().splitlines()
    for line in lines:
        line = line.split("%")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            p[k.strip().lower()] = v.strip()
    pts = []
    for n, line in enumerate(lines):
        if line.split("=")[0].strip().lower() == "npoints":
            pts = [l.split()[2] for l in lines[n + 1:n + 1 + int(line.split("=")[1])]]
    return p, pts


def detrend(a, t):
    """Remove a linear trend in time (the tide ramp) from a (nt, ...) array."""
    A = np.vstack([np.ones_like(t), t - t.mean()]).T
    coef = np.linalg.lstsq(A, a.reshape(len(t), -1), rcond=None)[0]
    return (a.reshape(len(t), -1) - A @ coef).reshape(a.shape)


def r4(x):
    return round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tspin", type=float, default=60.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    R, T = Path(a.ref), Path(a.test)
    p, pts = read_params(R)
    nx, ny, dx, tintg = int(p["nx"]), int(p["ny"]), float(p["dx"]), float(p["tintg"])
    tintp = float(p.get("tintp", tintg))
    zb = np.loadtxt(R / "z.grd")
    f = {}
    for v in ("zs", "u", "v"):
        r = np.fromfile(R / f"{v}.dat", "<f8").reshape(-1, ny + 1, nx + 1)
        s = np.fromfile(T / f"{v}.dat", "<f8").reshape(-1, ny + 1, nx + 1)
        m = min(len(r), len(s))
        f[v] = (r[:m], s[:m])
    m = f["zs"][0].shape[0]
    time = np.arange(m) * tintg
    sel = time >= a.tspin
    zr, zt = f["zs"][0][sel], f["zs"][1][sel]
    wet = ((zr - zb) > 0.05).all(axis=0)
    wet[[0, 1, ny - 1, ny], :] = False          # halo rows of the cyclic domain
    out = {"frames": int(m)}
    hr, ht = 4 * detrend(zr, time[sel]).std(axis=0), 4 * detrend(zt, time[sel]).std(axis=0)
    dh = (ht - hr)[wet]
    out["Hs_map"] = dict(mean_ref=r4(hr[wet].mean()), bias=r4(dh.mean()), rel_rmse=r4(np.sqrt((dh ** 2).mean()) / hr[wet].mean()))
    out["mean_level"] = dict(rmse_m=round(float(np.sqrt(((zt.mean(0) - zr.mean(0))[wet] ** 2).mean())), 5))
    ur, ut = f["u"][0][sel].mean(0), f["u"][1][sel].mean(0)
    vr, vt = f["v"][0][sel].mean(0), f["v"][1][sel].mean(0)
    du = np.hypot(ut - ur, vt - vr)[wet]
    sr = np.hypot(ur, vr)[wet]
    out["mean_current"] = dict(rms_ref=r4(np.sqrt((sr ** 2).mean())), rel_rmse=r4(np.sqrt((du ** 2).mean()) / np.sqrt((sr ** 2).mean())))
    rows = slice(2, ny - 1)                      # unique rows 3..ny-1 (1-based)
    Vr, Vt = vr[rows].mean(0), vt[rows].mean(0)
    out["longshore_V_profile"] = dict(max_ref=r4(np.abs(Vr).max()), max_test=r4(np.abs(Vt).max()),
                                      rmse=r4(np.sqrt(((Vt - Vr) ** 2).mean())), rel_rmse=r4(np.sqrt(((Vt - Vr) ** 2).mean()) / np.sqrt((Vr ** 2).mean())))
    out["zs_field"] = dict(rms_diff=round(float(np.sqrt(np.mean((zt - zr)[:, wet] ** 2))), 5),
                           rms_ref=round(float(detrend(zr, time[sel])[:, wet].std()), 5))
    out["seam_max_abs_zs_row1_minus_row_ny-2"] = dict(ref=r4(np.abs(f["zs"][0][:, 0] - f["zs"][0][:, ny - 3]).max()),
                                                     test=r4(np.abs(f["zs"][1][:, 0] - f["zs"][1][:, ny - 3]).max()))
    out["points"] = {}
    for name in pts:
        pr, pt = np.fromfile(R / f"{name}.dat", "<f8").reshape(-1, 6), np.fromfile(T / f"{name}.dat", "<f8").reshape(-1, 6)
        k = min(len(pr), len(pt))
        pr, pt = pr[:k], pt[:k]
        s = pr[:, 0] >= a.tspin
        res = dict(t_match=bool(np.allclose(pr[:, 0], pt[:, 0])))
        for col, v in ((1, "zs"), (4, "u"), (5, "v")):
            x, y = pr[s, col], pt[s, col]
            res[v + "_corr"] = r4(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else None
            res[v + "_rms_diff"] = r4(np.sqrt(np.mean((x - y) ** 2)))
        out["points"][name] = res
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
