"""
Feature-test deck generator: one small XBeach non-hydrostatic deck per option, to run unchanged in Fortran
XBeach and in xbnh_gpu.

Base case: barred beach, offshore depth 8 m, 1:50 slope up to +2 m, a shore-parallel bar at x = 300 m and, in
2D, a rip channel every 300 m alongshore. Irregular waves (JONSWAP Hm0 1 m, Tp 8 s) enter through a scalar
boun_U.bcf (wbctype = ts_nonh) unless --wbc selects another boundary type.

Examples:
  python make_case.py deck --ny 0                      1D flume (superfast 1D)
  python make_case.py deck --ny 0 --set nonhq3d=1      1D reduced two-layer
  python make_case.py deck --grid vardx                variable cross-shore spacing (xfile/yfile)
  python make_case.py deck --grid curvi                curvilinear grid
  python make_case.py deck --grid rot --set alfa=30    rotated uniform grid
  python make_case.py deck --tide loc2land             tideloc = 2 (paulrevere land)
  python make_case.py deck --veg drag                  two vegetation species (one with Cd < 0)
  python make_case.py deck --veg canopy                porous in-canopy model (porcanflow = 1)
  python make_case.py deck --ship track --wbc off      ship sailing alongshore (ships = 1)
  python make_case.py deck --sed basic                 sediment transport and morphology
  python make_case.py deck --sed basic --ridge         ... with a steep ridge that avalanches
Any XBeach key can be added or overridden with --set key=value (repeatable).
"""
import argparse
from pathlib import Path

import numpy as np

G = 9.81


def dispersion(w, h):
    k = w * w / G
    for _ in range(100):
        k = k - (G * k * np.tanh(k * h) - w * w) / (G * np.tanh(k * h) + G * k * h / np.cosh(k * h) ** 2)
    return k


def profile(x, y, ny, rip=True, ridge=False):
    zb = -8.0 + x / 50.0
    bar = 1.5 * np.exp(-((x - 300.0) / 25.0) ** 2)
    gap = 1.0 - np.exp(-((np.mod(y, 300.0) - 150.0) / 30.0) ** 4) if (ny > 0 and rip) else 1.0
    if ridge:   # submerged ridge at 4 m depth with flanks of slope 0.17 (above the default wetslp 0.15), wavy alongshore
        x0 = 200.0 + (10.0 * np.sin(2 * np.pi * y / 120.0) if ny > 0 else 0.0)
        zb = zb + 0.8 * np.exp(-((x - x0) / 4.0) ** 2)
    return np.minimum(zb + bar * gap, 2.0)


def wave_series(tstop, h0, hm0=1.0, tp=8.0, gamma=3.3, ncomp=200, seed=7, dt=0.1, ramp=20.0):
    f = np.linspace(0.05, 0.30, ncomp)
    df = f[1] - f[0]
    fp = 1.0 / tp
    sig = np.where(f <= fp, 0.07, 0.09)
    S = f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** np.exp(-((f - fp) ** 2) / (2 * sig ** 2 * fp ** 2))
    S *= (hm0 / 4.0) ** 2 / (S.sum() * df)
    a = np.sqrt(2 * S * df)
    w = 2 * np.pi * f
    k = dispersion(w, h0)
    phi = np.random.default_rng(seed).uniform(0, 2 * np.pi, ncomp)
    t = np.arange(0.0, tstop + 10.0 + dt / 2, dt)
    arg = np.outer(t, w) + phi
    z = (a * np.cos(arg)).sum(axis=1)
    u = (a * w / (k * h0) * np.cos(arg)).sum(axis=1)
    r = np.where(t < ramp, 0.5 - 0.5 * np.cos(np.pi * t / ramp), 1.0)
    return t, u * r, z * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--nx", type=int, default=0)
    ap.add_argument("--ny", type=int, default=60)
    ap.add_argument("--dx", type=float, default=2.0)
    ap.add_argument("--dy", type=float, default=0.0, help="default: dx")
    ap.add_argument("--tstop", type=float, default=120.0)
    ap.add_argument("--tintg", type=float, default=1.0)
    ap.add_argument("--grid", default="uniform", choices=["uniform", "vardx", "curvi", "rot"])
    ap.add_argument("--wbc", default="ts_nonh", choices=["ts_nonh", "params", "reuse", "off"])
    ap.add_argument("--tide", default="zs0", choices=["zs0", "loc1", "loc2land", "loc2sea", "loc4"])
    ap.add_argument("--veg", default="none", choices=["none", "drag", "canopy"])
    ap.add_argument("--ship", default="none", choices=["none", "track", "heading", "force"])
    ap.add_argument("--sed", default="none", choices=["none", "basic", "struct", "ngd2", "setbathy"])
    ap.add_argument("--norip", action="store_true")
    ap.add_argument("--ridge", action="store_true", help="add a steep submerged ridge (triggers avalanching)")
    ap.add_argument("--points", action="store_true", help="add point output and runup gauges")
    ap.add_argument("--means", action="store_true", help="add mean output (zs, u, v)")
    ap.add_argument("--extravars", default="", help="extra global output variables, comma separated")
    ap.add_argument("--set", action="append", default=[], help="key=value added to params.txt")
    a = ap.parse_args()
    dx = a.dx
    dy = a.dy or dx
    nx = a.nx or int(round(520.0 / dx))
    ny = a.ny
    out = Path(a.case)
    out.mkdir(parents=True, exist_ok=True)
    keys = {}
    # grid
    if a.grid == "vardx":
        # cross-shore spacing from 3 m offshore to 1 m nearshore, smooth
        s = np.linspace(0, 1, nx + 1)
        ds = 3.0 - 2.0 * (0.5 - 0.5 * np.cos(np.pi * np.clip(s * 1.3, 0, 1)))
        xs = np.concatenate([[0], np.cumsum(ds[:-1])])
        xs *= 520.0 / xs[-1]
        ys = np.arange(ny + 1) * dy
        X, Y = np.meshgrid(xs, ys)
        keys.update(vardx=1, xfile="x.grd", yfile="y.grd")
    elif a.grid == "curvi":
        xs = np.arange(nx + 1) * dx
        ys = np.arange(ny + 1) * dy
        X0, Y0 = np.meshgrid(xs, ys)
        # gently curved coast: grid lines bend by up to 4 % of the alongshore length
        L = max(ny, 1) * dy
        bend = 0.04 * L * np.sin(np.pi * Y0 / max(L, 1)) * (X0 / X0.max())
        X, Y = X0 + bend, Y0 + 0.02 * X0 * np.sin(2 * np.pi * Y0 / max(L, 1))
        keys.update(vardx=1, xfile="x.grd", yfile="y.grd")
    else:
        xs = np.arange(nx + 1) * dx
        ys = np.arange(ny + 1) * dy
        X, Y = np.meshgrid(xs, ys)
        keys.update(vardx=0, dx=dx, dy=dy)
    # bed from the local grid coordinates (cross-shore distance along grid lines)
    sx = np.zeros_like(X)
    sx[:, 1:] = np.cumsum(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)), axis=1)
    zb = profile(sx, Y if a.grid != "curvi" else np.meshgrid(np.arange(nx + 1), np.arange(ny + 1) * dy)[1], ny, rip=not a.norip, ridge=a.ridge)
    np.savetxt(out / "z.grd", np.atleast_2d(zb), fmt="%.4f")
    if keys.get("vardx") == 1:
        np.savetxt(out / "x.grd", np.atleast_2d(X), fmt="%.6f")
        np.savetxt(out / "y.grd", np.atleast_2d(Y), fmt="%.6f")
    h0 = -float(np.atleast_2d(zb)[0, 0])
    # waves
    if a.wbc in ("ts_nonh", "reuse"):
        t, u, z = wave_series(a.tstop, h0)
        name = "boun_U.bcf" if a.wbc == "ts_nonh" else "nh_series00001.bcf"
        with open(out / name, "w") as fh:
            fh.write("scalar\n3\nt u z\n")
            for row in zip(t, u, z):
                fh.write("%.3f %.6f %.6f\n" % row)
        if a.wbc == "reuse":
            (out / "nhbcflist.bcf").write_text(f"0.0 {t[-1]:.1f} 8.0 0.7 nh_series00001.bcf\n")
        keys.update(wbctype=a.wbc)
    elif a.wbc == "params":
        keys.update(wbctype="params", Hrms=0.6, Trep=8.0, dir0=270)
    else:
        keys.update(wbctype="off")
    # tide
    tt = np.array([0.0, a.tstop / 2, a.tstop + 50])
    if a.tide == "zs0":
        keys.update(zs0=0.0)
    elif a.tide == "loc1":
        np.savetxt(out / "tide.txt", np.c_[tt, [0.0, 0.3, 0.1]], fmt="%.3f")
        keys.update(zs0file="tide.txt", tideloc=1)
    elif a.tide in ("loc2land", "loc2sea"):
        np.savetxt(out / "tide.txt", np.c_[tt, [0.0, 0.3, 0.1], [0.0, -0.1, 0.05]], fmt="%.3f")
        keys.update(zs0file="tide.txt", tideloc=2, paulrevere=a.tide[4:])
    else:
        np.savetxt(out / "tide.txt", np.c_[tt, [0.0, 0.3, 0.1], [0.0, 0.2, 0.0], [0.0, -0.1, 0.05], [0.0, 0.1, 0.1]], fmt="%.3f")
        keys.update(zs0file="tide.txt", tideloc=4)
    # vegetation: species 1 on x = 380..430 m, species 2 on 430..470 m, over the first 60 % of the alongshore length
    if a.veg != "none":
        L = max(ny, 1) * dy
        band = (Y <= 0.6 * L) if ny > 0 else np.ones_like(X, bool)
        vt = np.where(band & (sx >= 380) & (sx < 430), 1, 0) + np.where(band & (sx >= 430) & (sx < 470), 2, 0)
        if a.veg == "drag":
            spec = {"spec1.txt": ["nsec = 2", "ah = 0.4 1.0", "bv = 0.01 0.005", "N = 400 200", "Cd = 1.0 0.8"],
                    "spec2.txt": ["nsec = 1", "ah = 0.6", "bv = 0.02", "N = 250", "Cd = -1"]}
        else:
            spec = {"spec1.txt": ["isCanopy = 1", "nsec = 1", "ah = 0.5", "Kp_can = 0.001", "beta_can = 15",
                                  "cf_can = 0.01", "cm_can = 1.5", "por_can = 0.8"]}
            vt = np.where(vt > 0, 1, 0)
            keys.update(porcanflow=1)
        for fn, lines_ in spec.items():
            (out / fn).write_text("\n".join(lines_) + "\n")
        (out / "veggiefile.txt").write_text("".join(fn + "\n" for fn in spec))
        np.savetxt(out / "veggiemap.txt", np.atleast_2d(vt), fmt="%d")
        keys.update(vegetation=1, veggiefile="veggiefile.txt", veggiemapfile="veggiemap.txt")
    # ship: 40 m x 16 m hull, 1.5 m draft, sailing alongshore at x = 150 m (about 5 m depth)
    if a.ship != "none":
        L, Bw = 40.0, 16.0
        sx_, sy_ = 2.0, 2.0
        snx, sny = int(L / sx_), int(Bw / sy_)
        xs_ = (np.arange(snx + 1) - snx / 2) * sx_
        ys_ = (np.arange(sny + 1) - sny / 2) * sy_
        XX, YY = np.meshgrid(xs_, ys_)
        hull = np.clip(1.5 * (1 - (2 * XX / L) ** 2) * (1 - (2 * YY / Bw) ** 2), 0, None)
        np.savetxt(out / "shipgeom.txt", hull, fmt="%.4f")
        tt = np.array([0.0, a.tstop + 20])
        Ly = max(ny, 1) * dy
        if a.ship == "track":
            np.savetxt(out / "track.txt", np.c_[tt, [150.0, 150.0], [-10.0, Ly + 10.0]], fmt="%.3f")
            extra = ""
        else:   # prescribed heading, slightly oblique track
            np.savetxt(out / "track.txt", np.c_[tt, [140.0, 170.0], [-10.0, Ly + 10.0], [0.0, 0.0]], fmt="%.3f")
            extra = "read_heading = 1\n"
        (out / "ship1.txt").write_text(f"dx = {sx_}\ndy = {sy_}\nnx = {snx}\nny = {sny}\nshipgeom = shipgeom.txt\n"
                                       f"shiptrack = track.txt\ncompute_force = {1 if a.ship == 'force' else 0}\n" + extra)
        (out / "ships.txt").write_text("ship1.txt\n")
        keys.update(ships=1, shipfile="ships.txt")
    # sediment: sedtrans + morphology (avalanching on by default); struct: non-erodible layer 2 mm below the bar (exposed
    # within a 120 s run);
    # ngd2: two fractions (fine offshore, coarse nearshore) in three bed layers; setbathy: two prescribed beds
    if a.sed != "none":
        keys.update(sedtrans=1, morphology=1)
        if a.sed == "struct":
            ne = np.where((sx > 250) & (sx < 350), 0.002, 10.0)
            np.savetxt(out / "ne_layer.txt", np.atleast_2d(ne), fmt="%.3f")
            keys.update(struct=1, ne_layer="ne_layer.txt")
        if a.sed == "ngd2":
            f1 = np.clip((400.0 - sx) / 200.0, 0.2, 0.8)
            for g, fr in ((1, f1), (2, 1 - f1)):
                with open(out / f"gdist{g}.inp", "w") as fh:
                    for _ in range(3):
                        np.savetxt(fh, np.atleast_2d(fr), fmt="%.4f")
            keys.update(ngd=2, nd=3, D50="0.0002 0.0005", D90="0.0003 0.0008")
        if a.sed == "setbathy":
            keys.pop("sedtrans"); keys.pop("morphology")
            with open(out / "bathyseries.txt", "w") as fh:
                for tt_, dz in ((0.0, 0.0), (a.tstop, 0.3)):
                    fh.write(f"{tt_:.1f}\n")
                    np.savetxt(fh, np.atleast_2d(zb + dz * np.exp(-((sx - 300.0) / 30.0) ** 2)), fmt="%.4f")
            keys.update(setbathy=1, nsetbathy=2, setbathyfile="bathyseries.txt")
    keys.update(nx=nx, ny=ny, depfile="z.grd", posdwn=-1, wavemodel="nonh", tstart=0, tstop=a.tstop,
                tintg=a.tintg, outputformat="fortran", outputprecision="double", back="wall", left="neumann",
                right="neumann")
    for s in a.set:
        k, v = s.split("=", 1)
        keys[k.strip()] = v.strip()
    gvars = ["zs", "u", "v", "zb"] + [v for v in a.extravars.split(",") if v]
    lines = ["%% xbnh-gpu feature-test deck, examples/features/make_case.py"]
    lines += [f"{k:<14}= {v}" for k, v in keys.items()]
    if keys.get("outputformat") == "fortran":
        lines.append("outputformat  = fortran")   # trunk r6167 needs the statement twice (snappointstogrid format)
    lines.append(f"nglobalvar    = {len(gvars)}")
    lines += gvars
    if a.points:
        yp = 0.5 * ny * dy
        pts = [(100.0, yp, "off"), (300.0, yp, "bar"), (360.0, yp, "trough")]
        lines.append(f"npoints       = {len(pts)}")
        lines += [f"{x:.1f} {y:.1f} {nm}" for x, y, nm in pts]
        lines.append("npointvar     = 4")
        lines += ["zs", "u", "v", "hh"]
        lines.append("nrugauge      = 1")
        lines.append(f"0.0 {yp:.1f} rug1")
        lines.append("tintp         = 0.5")
    if a.means:
        lines += ["nmeanvar      = 3", "zs", "u", "v", f"tintm         = {a.tstop / 2}"]
    (out / "params.txt").write_text("\n".join(lines) + "\n")
    print(f"{out}: nx={nx} ny={ny} grid={a.grid} h0={h0:.2f}")


if __name__ == "__main__":
    main()
