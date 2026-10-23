"""
Compare two runs of the same XBeach deck (Fortran-format output of XBeach or xbnh_gpu), 1D or 2D.

Statistics over t >= tspin of the global output:
  Hs_rel     RMS difference of Hs = 4 std(zs) over wet cells, relative to the mean reference Hs
  current_rel  RMS difference of the mean velocity vector (<u>, <v>), relative to the RMS reference current
  setup_m    RMS difference of the mean water level [m]
  zs_rms_m   RMS difference of the instantaneous zs field [m]
Point output (if present): RMS difference of each point variable.

Usage: python tools/compare_runs.py <ref dir> <test dir> [--tspin 60] [--vars zs u v] [--json out.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_params(d):
    p, lines = {}, (Path(d) / "params.txt").read_text().splitlines()
    for line in lines:
        line = line.split("%")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            if k not in p:
                p[k] = v.strip()
    return p, lines


def list_after(lines, key, n):
    for q, line in enumerate(lines):
        if "=" in line and line.split("=")[0].strip().lower() == key:
            return [l.split()[0] if key != "npoints" else l.split() for l in lines[q + 1:q + 1 + n]]
    return []


def load(d, var, n):
    f = Path(d) / f"{var}.dat"
    if not f.exists():
        return None
    a = np.fromfile(f, "<f8")
    return a[: (a.size // n) * n].reshape(-1, n)


def stats(R, T, tspin):
    p, lines = read_params(R)
    nx, ny = int(p["nx"]), int(p["ny"])
    tintg = float(p.get("tintg", p.get("tint", 1)))
    n = (nx + 1) * (ny + 1)
    out = {}
    zr, zt = load(R, "zs", n), load(T, "zs", n)
    m = min(len(zr), len(zt))
    zr, zt = zr[:m], zt[:m]
    time = np.arange(m) * tintg
    sel = time >= tspin
    zbf = load(R, "zb", n)
    if zbf is None:
        dep = p.get("depfile")
        zb = np.loadtxt(Path(R) / dep).reshape(-1) * (-1 if float(p.get("posdwn", 1)) > 0 else 1) if dep else zr[0] * 0 - 100
    else:
        zb = zbf[0]
    wet = ((zr[sel] - zb) > 0.05).all(axis=0)
    if ny > 0:
        g = wet.reshape(ny + 1, nx + 1)
        g[[0, ny], :] = False
        g[:, [0, nx]] = False
        wet = g.reshape(-1)
    else:
        wet[[0, nx]] = False
    if not (np.isfinite(zt).all()):
        return {"finite": False}
    hr, ht = 4 * zr[sel].std(axis=0), 4 * zt[sel].std(axis=0)
    out["Hs_mean_ref"] = float(hr[wet].mean())
    out["Hs_rel"] = float(np.sqrt(((hr - ht)[wet] ** 2).mean()) / max(hr[wet].mean(), 1e-12))
    out["setup_m"] = float(np.sqrt(((zr[sel].mean(0) - zt[sel].mean(0))[wet] ** 2).mean()))
    out["zs_rms_m"] = float(np.sqrt(((zr - zt)[sel][:, wet] ** 2).mean()))
    ur, ut = load(R, "u", n), load(T, "u", n)
    vr, vt = load(R, "v", n), load(T, "v", n)
    if ur is not None and ut is not None:
        mu_r, mu_t = ur[:m][sel].mean(0), ut[:m][sel].mean(0)
        mv_r = vr[:m][sel].mean(0) if vr is not None and vt is not None else 0 * mu_r
        mv_t = vt[:m][sel].mean(0) if vr is not None and vt is not None else 0 * mu_t
        rms_ref = np.sqrt((mu_r[wet] ** 2 + mv_r[wet] ** 2).mean())
        out["current_rms_ref"] = float(rms_ref)
        out["current_rel"] = float(np.sqrt(((mu_r - mu_t)[wet] ** 2 + (mv_r - mv_t)[wet] ** 2).mean()) / max(rms_ref, 1e-12))
        out["u_rms"] = float(np.sqrt(((ur[:m] - ut[:m])[sel][:, wet] ** 2).mean()))
    # points
    npts = int(p.get("npoints", 0))
    if npts > 0:
        names = [l[2] if len(l) > 2 else "point%03d" % (q + 1) for q, l in enumerate(list_after(lines, "npoints", npts))]
        nv = int(p.get("npointvar", 0))
        pv = list_after(lines, "npointvar", nv)
        if int(p.get("nrugauge", 0)) > 0:   # XBeach adds xz, yz, zs for runup gauges
            pv += [v for v in ("xz", "yz", "zs") if v not in pv]
            nv = len(pv)
        pts = {}
        for nm in names:
            a = np.fromfile(Path(R) / f"{nm}.dat", "<f8").reshape(-1, nv + 1)
            b = np.fromfile(Path(T) / f"{nm}.dat", "<f8").reshape(-1, nv + 1)
            k = min(len(a), len(b))
            s = a[:k, 0] >= tspin
            pts[nm] = {"t_match": bool(np.allclose(a[:k, 0], b[:k, 0])),
                       **{v: float(np.sqrt(((a[:k, q + 1] - b[:k, q + 1])[s] ** 2).mean())) for q, v in enumerate(pv)}}
        out["points"] = pts
    out["frames"] = int(m)
    out["finite"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tspin", type=float, default=60.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    s = stats(a.ref, a.test, a.tspin)
    txt = json.dumps(s, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
