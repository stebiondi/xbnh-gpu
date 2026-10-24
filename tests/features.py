"""
Feature test suite (tests/features/suite.txt): one small deck per XBeach option.

  features.py make <work dir> [names]            build the decks (<work>/<name>/deck)
  features.py compare <work dir> [names]         GPU run (<name>/gpu) against Fortran XBeach (<name>/xb)
  features.py ref <work dir> [names]             GPU run against the XBeach statistics in tests/features/reference.json
  features.py refstats <work dir>                write tests/features/reference.json from the XBeach runs
  features.py list                               print the deck names
Options: --run <folder> (default gpu, e.g. gpu_sp), --json <file> (write the table as JSON).

compare: differences of the global output over t >= tspin (wet cells away from the boundaries):
  Hs_rel       RMS difference of Hs = 4 std(zs), relative to the mean reference Hs
  current_rel  RMS difference of the mean velocity vector, relative to the RMS reference current
  zs_rms_m     RMS difference of the instantaneous water level [m]
  bed_rel      RMS difference of the final bed, relative to the RMS reference bed change (morphology decks)
  cc_rel       RMS difference of the sediment concentration, relative to its RMS (when ccg is written)
  files_rel    largest difference in any other output file (extra global variables, points, drifters, ships),
               relative to the largest reference value in that file
The two runs must have the same .dat filenames and file sizes before their values are compared.
ref: relative differences of the deck statistics (mean Hs, RMS mean current, mean water level, RMS bed change,
mean concentration) from the values of XBeach r6167 serial, for machines without a Fortran XBeach build.

1D decks agree with XBeach to round-off (1e-13 or better in our tests), so their limits are 1e-6. 2D decks
agree to the tolerance of the iterative pressure solvers (1e-4 relative at the default tolerances), so their
limits are 2e-3 in Hs and 1e-2 in mean current, bed and concentration.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "features" / "suite.txt"
INPUTS = ROOT / "tests" / "features" / "inputs"
REFJSON = ROOT / "tests" / "features" / "reference.json"
LIMITS = {"1d": {"Hs_rel": 1e-6, "current_rel": 1e-6, "zs_rms_m": 1e-6, "bed_rel": 1e-6, "cc_rel": 1e-6, "files_rel": 1e-6},
          "2d": {"Hs_rel": 2e-3, "current_rel": 1e-2, "zs_rms_m": 5e-3, "bed_rel": 1e-2, "cc_rel": 1e-2, "files_rel": 1e-1}}
INTVARS = {"wetz", "wetu", "wetv", "wete", "breaking"}
STAT_LIMITS = {"1d": {"Hs_mean": 1e-6, "current_rms": 1e-6, "zs_mean": 1e-6, "dzb_rms": 1e-6, "cc_mean": 1e-6},
               "2d": {"Hs_mean": 2e-3, "current_rms": 1e-2, "zs_mean": 5e-3, "dzb_rms": 1e-2, "cc_mean": 1e-2}}


def suite():
    out = []
    for line in SUITE.read_text().splitlines():
        line = line.split("#")[0].split()
        if line:
            out.append((line[0], line[1], [t for t in line[2:] if not t.startswith("+")],
                        [t[1:] for t in line[2:] if t.startswith("+")]))
    return out


def select(names):
    s = suite()
    if not names:
        return s
    unknown = set(names) - {c[0] for c in s}
    if unknown:
        sys.exit(f"unknown deck(s): {' '.join(sorted(unknown))}")
    return [c for c in s if c[0] in names]


def read_params(d):
    p, lines = {}, (Path(d) / "params.txt").read_text().splitlines()
    for line in lines:
        line = line.split("%")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            p.setdefault(k.strip().lower(), v.strip())
    return p, lines


class Run:
    """Fortran-format global output of one run (XBeach or xbnh_gpu)."""

    def __init__(self, d, deck=None):
        self.d = Path(d)
        p, self.lines = read_params(deck or d)
        self.nx, self.ny = int(p["nx"]), int(p["ny"])
        self.n = (self.nx + 1) * (self.ny + 1)
        if "tsglobal" in p:
            v = np.loadtxt(Path(deck or d) / p["tsglobal"])
            self.times = v[1:1 + int(v[0])]
        else:
            t0, t1, dt = float(p.get("tstart", 0)), float(p["tstop"]), float(p.get("tintg", 1))
            self.times = t0 + dt * np.arange(int(np.floor((t1 - t0) / dt + 1e-9)) + 1)
        self.tspin = min(40.0, float(p["tstop"]) / 3)
        zb0 = np.loadtxt(Path(deck or d) / p["depfile"]).reshape(-1)
        # statistics mask: cells always submerged (initial bed below -0.5 m), away from the grid boundaries
        m = (zb0 < -0.5).reshape(self.ny + 1, self.nx + 1)
        m[:, [0, -1]] = False
        if self.ny > 0:
            m[[0, 1, -2, -1], :] = False
        self.mask = m.reshape(-1)

    def load(self, var):
        f = self.d / f"{var}.dat"
        if not f.exists():
            return None
        a = np.fromfile(f, "<f8")
        return a[: (a.size // self.n) * self.n].reshape(-1, self.n)

    def sel(self, a):
        k = min(len(a), len(self.times))
        return a[:k][self.times[:k] >= self.tspin]


def stats(run):
    """Deck statistics (stored for XBeach in reference.json)."""
    out = {}
    zs = run.load("zs")
    if zs is None or not np.isfinite(zs).all():
        return {"finite": False}
    s = run.sel(zs)
    m = run.mask
    out["Hs_mean"] = float((4 * s.std(axis=0))[m].mean())
    out["zs_mean"] = float(s.mean(axis=0)[m].mean())
    u, v = run.load("u"), run.load("v")
    if u is not None:
        mu = run.sel(u).mean(axis=0)
        mv = run.sel(v).mean(axis=0) if v is not None else 0 * mu
        out["current_rms"] = float(np.sqrt((mu[m] ** 2 + mv[m] ** 2).mean()))
    zb = run.load("zb")
    if zb is not None and len(zb) > 1:
        out["dzb_rms"] = float(np.sqrt(((zb[-1] - zb[0]) ** 2).mean()))
    cc = run.load("ccg")
    if cc is not None:   # all frames and fractions
        out["cc_mean"] = float(cc.mean())
    out["finite"] = True
    return out


def compare(ref, test):
    """Differences between two runs of the same deck."""
    R, T = ref, test
    ref_files = {f.name: f for f in R.d.glob("*.dat")}
    test_files = {f.name: f for f in T.d.glob("*.dat")}
    errors = [f"missing output: {name}" for name in sorted(ref_files.keys() - test_files.keys())]
    errors += [f"unexpected output: {name}" for name in sorted(test_files.keys() - ref_files.keys())]
    for name in sorted(ref_files.keys() & test_files.keys()):
        expected, actual = ref_files[name].stat().st_size, test_files[name].stat().st_size
        if actual != expected:
            errors.append(f"{name}: {actual} bytes, expected {expected}")
    if errors:
        return {"finite": False, "errors": errors}
    zr, zt = R.load("zs"), T.load("zs")
    if zt is None or not np.isfinite(zt).all():
        return {"finite": False}
    k = min(len(zr), len(zt))
    zr, zt = R.sel(zr[:k]), R.sel(zt[:k])
    m = R.mask
    out = {}
    hr, ht = 4 * zr.std(axis=0), 4 * zt.std(axis=0)
    out["Hs_rel"] = float(np.sqrt(((hr - ht)[m] ** 2).mean()) / max(hr[m].mean(), 1e-12))
    out["zs_rms_m"] = float(np.sqrt(((zr - zt)[:, m] ** 2).mean()))
    ur, ut, vr, vt = R.load("u"), T.load("u"), R.load("v"), T.load("v")
    if ur is not None and ut is not None:
        mur, mut = R.sel(ur[:k]).mean(0), R.sel(ut[:k]).mean(0)
        mvr = R.sel(vr[:k]).mean(0) if vr is not None and vt is not None else 0 * mur
        mvt = R.sel(vt[:k]).mean(0) if vr is not None and vt is not None else 0 * mut
        rms = np.sqrt((mur[m] ** 2 + mvr[m] ** 2).mean())
        out["current_rel"] = float(np.sqrt(((mur - mut)[m] ** 2 + (mvr - mvt)[m] ** 2).mean()) / max(rms, 1e-12))
    br, bt = R.load("zb"), T.load("zb")
    if br is not None and bt is not None and len(br) > 1:
        ch = np.sqrt(((br[-1] - br[0]) ** 2).mean())
        if ch > 1e-6:
            out["bed_rel"] = float(np.sqrt(((br[min(len(br), len(bt)) - 1] - bt[min(len(br), len(bt)) - 1]) ** 2).mean()) / ch)
    cr, ct = R.load("ccg"), T.load("ccg")
    if cr is not None and ct is not None:
        kk = min(len(cr), len(ct))
        a, b = cr[:kk], ct[:kk]
        rms = np.sqrt((a ** 2).mean())
        if rms > 0:
            out["cc_rel"] = float(np.sqrt(((a - b) ** 2).mean()) / rms)
    # every other output file (global variables, points, runup gauges, drifters, per-ship variables): largest
    # difference relative to the largest reference value, fill values excluded
    worst = 0.0
    for f in sorted(R.d.glob("*.dat")):
        if f.name in ("xy.dat", "dims.dat", "zs.dat", "u.dat", "v.dat", "zb.dat", "ccg.dat") or f.stem in INTVARS:
            continue
        g = T.d / f.name
        if not g.exists():
            worst = np.inf
            continue
        a, b = np.fromfile(f, "<f8"), np.fromfile(g, "<f8")
        kk = min(a.size, b.size)
        a, b = a[:kk], b[:kk]
        ok = (np.abs(a) < 1e300) & (np.abs(a) > 1e-300)
        ok |= (a == 0)
        scale = np.abs(a[ok]).max() if ok.any() else 0
        if scale > 0:
            worst = max(worst, float(np.abs(a[ok] - b[ok]).max() / scale))
    out["files_rel"] = worst
    out["frames"] = int(k)
    out["finite"] = True
    return out


def judge(vals, lim):
    bad = list(vals.get("errors", []))
    bad += [key for key, v in vals.items() if key in lim and not (np.isfinite(v) and v <= lim[key])]
    return (not bad) and vals.get("finite", True), bad


def cmd_make(work, names):
    mk = ROOT / "examples" / "features" / "make_case.py"
    for name, _, args, files in select(names):
        deck = Path(work) / name / "deck"
        if deck.exists():
            shutil.rmtree(deck)
        subprocess.run([sys.executable, str(mk), str(deck)] + args, check=True, capture_output=True)
        for f in files:
            shutil.copy(INPUTS / f, deck / f)
    print(f"built {len(select(names))} decks in {work}")


def report(rows, title, jsonfile=None):
    if jsonfile:
        Path(jsonfile).write_text(json.dumps({"title": title, "decks": {name: {"class": cls, "pass": bool(ok), **vals,
                                                                          **({"errors": bad} if bad else {})}
                                                                          for name, cls, vals, ok, bad in rows}}, indent=1) + "\n")
    print(title)
    npass = 0
    for name, cls, vals, ok, bad in rows:
        txt = "  ".join(f"{k} {v:.1e}" for k, v in vals.items() if isinstance(v, float))
        print(f"  {name:13s} {cls}  {'PASS' if ok else 'FAIL'}  {txt}" + (f"   (failed checks: {', '.join(bad)})" if bad else ""))
        npass += ok
    print(f"features passed: {npass} of {len(rows)}")
    return npass == len(rows)


def cmd_compare(work, names, run="gpu", jsonfile=None):
    rows = []
    for name, cls, _, _ in select(names):
        base = Path(work) / name
        if not (base / "xb" / "zs.dat").exists() or not (base / run / "zs.dat").exists():
            rows.append((name, cls, {}, False, ["missing run"]))
            continue
        vals = compare(Run(base / "xb", base / "deck"), Run(base / run, base / "deck"))
        ok, bad = judge(vals, LIMITS[cls])
        rows.append((name, cls, {k: v for k, v in vals.items() if k in LIMITS[cls]}, ok, bad))
    return report(rows, f"{run} against Fortran XBeach", jsonfile)


def cmd_ref(work, names, run="gpu", jsonfile=None):
    ref = json.loads(REFJSON.read_text())["decks"]
    rows = []
    for name, cls, _, _ in select(names):
        base = Path(work) / name
        if name not in ref or not (base / run / "zs.dat").exists():
            rows.append((name, cls, {}, False, ["missing run or reference"]))
            continue
        s = stats(Run(base / run, base / "deck"))
        vals, bad = {}, []
        for key, lim in STAT_LIMITS[cls].items():
            if key not in ref[name]:
                continue
            r, t = ref[name][key], s.get(key, np.nan)
            scale = max(abs(r), 1e-3 if key == "zs_mean" else 1e-12)   # zs_mean: absolute difference [m]
            vals[key] = abs(t - r) / (scale if key != "zs_mean" else 1.0)
            if not (np.isfinite(vals[key]) and vals[key] <= lim):
                bad.append(key)
        rows.append((name, cls, vals, s.get("finite", False) and not bad, bad))
    return report(rows, f"{run} against stored XBeach statistics (tests/features/reference.json)", jsonfile)


def cmd_refstats(work):
    decks = {}
    for name, cls, _, _ in suite():
        base = Path(work) / name
        if (base / "xb" / "zs.dat").exists():
            decks[name] = stats(Run(base / "xb", base / "deck"))
    REFJSON.write_text(json.dumps({"source": "XBeach r6167 serial (trunk), tests/features/suite.txt decks",
                                   "decks": decks}, indent=1) + "\n")
    print(f"wrote {REFJSON} ({len(decks)} decks)")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c, rest = sys.argv[1], sys.argv[2:]
    opt = {"--run": "gpu", "--json": None}
    for key in opt:
        if key in rest:
            q = rest.index(key)
            opt[key] = rest[q + 1]
            rest = rest[:q] + rest[q + 2:]
    run, jsonfile = opt["--run"], opt["--json"]
    if c == "list":
        print(" ".join(s[0] for s in suite()))
    elif c == "make":
        cmd_make(rest[0], rest[1:])
    elif c == "compare":
        sys.exit(0 if cmd_compare(rest[0], rest[1:], run, jsonfile) else 1)
    elif c == "ref":
        sys.exit(0 if cmd_ref(rest[0], rest[1:], run, jsonfile) else 1)
    elif c == "refstats":
        cmd_refstats(rest[0])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
