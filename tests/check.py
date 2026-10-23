"""
Pass/fail checks used by run_tests.sh.

  check.py rip <ref dir> <test dir>      rip-beach agreement (tools/compare_rip.py metrics)
  check.py fields <ref dir> <test dir>   cyclic-feature agreement (tools/compare_fields.py metrics)
  check.py ref <rip|cyc> <test dir>      GPU statistics against tests/reference.json (no XBeach needed)
  check.py hotstart <run dir> <restart dir>   state written by writehotstart is read back by hotstart = 1
  check.py same <dir a> <dir b>          zs, uu, vv equal to round-off (1e-9)

Thresholds are several times the differences found between XBeach builds and rank counts (see the guide),
so a pass means the GPU code reproduces XBeach, not that the two are identical.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {
    "rip": {"Hs_rel": 0.01, "current_rel": 0.03, "zs_rms_m": 0.01},
    "fields": {"Hs_rel": 0.02, "current_rel": 0.10, "zs_rms_m": 0.02},
}


def metrics(tool, ref, test, tspin):
    out = subprocess.run([sys.executable, str(ROOT / "tools" / tool), ref, test, "--tspin", str(tspin)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def report(vals, lim):
    ok = True
    for k, v in vals.items():
        good = np.isfinite(v) and v <= lim[k]
        ok &= bool(good)
        print(f"   {k:12s} {v:9.4f}  (limit {lim[k]})  {'ok' if good else 'TOO LARGE'}")
    return ok


def hs_mean(d, tspin):
    p = {}
    for line in (Path(d) / "params.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("%")[0].split("=", 1)
            p[k.strip().lower()] = v.strip()
    nx, ny, dt = int(p["nx"]), int(p["ny"]), float(p["tintg"])
    zs = np.fromfile(Path(d) / "zs.dat", "<f8").reshape(-1, ny + 1, nx + 1)
    zb = np.loadtxt(Path(d) / "z.grd")
    t = np.arange(len(zs)) * dt
    s = zs[t >= tspin]
    wet = ((s - zb) > 0.05).all(axis=0)
    wet[[0, 1, ny - 1, ny], :] = False
    if not np.isfinite(zs).all():
        return float("nan")
    tt = t[t >= tspin]
    a = s.reshape(len(tt), -1)
    A = np.vstack([np.ones_like(tt), tt - tt.mean()]).T
    a = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float((4 * a.std(axis=0)).reshape(ny + 1, nx + 1)[wet].mean())


def main():
    mode = sys.argv[1]
    if mode == "rip":
        m = metrics("compare_rip.py", sys.argv[2], sys.argv[3], 60)
        vals = {"Hs_rel": m["Hs_map"]["rel_rmse"], "current_rel": m["mean_current"]["rel_rmse"], "zs_rms_m": m["zs_field"]["rms_diff"]}
        ok = report(vals, LIMITS["rip"])
    elif mode == "fields":
        m = metrics("compare_fields.py", sys.argv[2], sys.argv[3], 40)
        vals = {"Hs_rel": m["Hs_map"]["rel_rmse"], "current_rel": m["mean_current"]["rel_rmse"], "zs_rms_m": m["zs_field"]["rms_diff"]}
        ok = report(vals, LIMITS["fields"])
    elif mode == "hotstart":
        # last global record of the writing run against the first record of the run restarted from it
        a, b = Path(sys.argv[2]), Path(sys.argv[3])
        p = {}
        for line in (a / "params.txt").read_text().splitlines():
            if "=" in line:
                k, v = line.split("%")[0].split("=", 1)
                p[k.strip().lower()] = v.strip()
        n = (int(p["nx"]) + 1) * (int(p["ny"]) + 1)
        ok = True
        for v in ["zs", "uu", "vv"]:
            x = np.fromfile(a / f"{v}.dat", "<f8").reshape(-1, n)[-1]
            y = np.fromfile(b / f"{v}.dat", "<f8").reshape(-1, n)[0]
            d = float(np.abs(x - y).max())
            good = d <= 1e-6
            ok &= good
            print(f"   {v:3s} max difference after restart {d:.1e}  (limit 1e-06)  {'ok' if good else 'TOO LARGE'}")
    elif mode == "same":
        # two runs that must agree to round-off (e.g. uniform bedfricfile against bedfriccoef)
        a, b = Path(sys.argv[2]), Path(sys.argv[3])
        ok = True
        for v in ["zs", "uu", "vv"]:
            x, y = np.fromfile(a / f"{v}.dat", "<f8"), np.fromfile(b / f"{v}.dat", "<f8")
            d = float(np.abs(x - y).max()) if x.shape == y.shape else float("inf")
            good = d <= 1e-9
            ok &= good
            print(f"   {v:3s} max difference {d:.1e}  (limit 1e-09, round-off)  {'ok' if good else 'TOO LARGE'}")
    else:
        case, d = sys.argv[2], sys.argv[3]
        ref = json.loads((ROOT / "tests" / "reference.json").read_text())[case]
        hs = hs_mean(d, ref["tspin"])
        rel = abs(hs - ref["Hs_mean_xbeach"]) / ref["Hs_mean_xbeach"]
        print(f"   mean Hs {hs:.4f} m, XBeach reference {ref['Hs_mean_xbeach']:.4f} m, difference {100 * rel:.2f} % (limit {100 * ref['limit']:.0f} %)")
        ok = np.isfinite(hs) and rel <= ref["limit"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
