# xbnh-gpu: XBeach non-hydrostatic on the GPU

A CUDA port of XBeach (Deltares, trunk r6167) in `wavemodel = nonh`, the phase-resolving
non-hydrostatic mode. Version 2 covers everything XBeach accepts in that mode: one-layer and reduced
two-layer flow, 1D and 2D grids (uniform, variable, rotated, curvilinear), all non-hydrostatic boundary
types, spectral wave boundaries, tide, wind, discharges and rainfall, vegetation, groundwater, ships,
sediment transport and morphology (sand and gravel formulations, XBeach-G), and Fortran and netCDF
output. It reads ordinary XBeach input (`params.txt` and the files it names) and writes XBeach output
files, so an existing XBeach-NH deck runs unchanged on one NVIDIA GPU.

- **Validated against Fortran XBeach**, option by option (96 test decks, `tests/features/suite.txt`).
  1D decks agree with XBeach to round-off (1e-13 or better). 2D decks agree to the tolerance of the
  iterative pressure solver (1e-4 in wave height at the default tolerances). With both solvers converged
  to 1e-11, 2D water levels and velocities agree to 1e-13 and sediment fields to 1e-14.
- **Speed:** on the rip-channel beach deck at 0.16 to 8.7 M cells, an RTX 2000 Ada laptop GPU ran
  2.0x to 3.3x (fp64) and 3.9x to 7.5x (fp32) faster than XBeach with MPI at its best rank count on a
  24-core workstation (i9-14900K), with the larger factors on the larger grids.

The full guide (installation, usage, options, validation, performance) is
**[docs/xbnh-gpu_guide.pdf](docs/xbnh-gpu_guide.pdf)**.

## Quick start (Linux or Windows WSL2, no admin rights)

You need an NVIDIA GPU with compute capability 6.0 or newer (Pascal and later), the NVIDIA driver (on
WSL2, the normal Windows driver) and an internet connection for the first install.

```bash
git clone https://github.com/stebiondi/xbnh-gpu xbnh-gpu && cd xbnh-gpu
./install.sh                    # micromamba + CUDA 12.6 toolchain in user space, builds bin/xbnh_gpu{,_sp}
source env.sh                   # puts xbnh_gpu on PATH (do this in every new shell)
bash tests/run_tests.sh         # quick tests, 2 to 5 minutes
bash tests/run_features.sh      # 96 feature decks against stored XBeach statistics, about 5 minutes
```

To also build the reference Fortran XBeach (serial and MPI) and compare against it directly:

```bash
./install.sh --with-xbeach      # adds gfortran + Open MPI in user space, fetches XBeach r6167, 10-20 min
bash tests/run_features.sh --xbeach -j 8    # runs every deck in XBeach as well, 20-60 min
```

## Running a model

```bash
cd my_xbeach_deck               # an XBeach-NH case: params.txt, bathymetry, boundary files
xbnh_gpu                        # double precision
xbnh_gpu_sp                     # single precision: about 2.2x faster
```

Run the GPU code in its own folder, because its output files have the same names as XBeach's. Output is
written as XBeach writes it: `xboutput.nc` (`outputformat = netcdf`) or the Fortran files (`zs.dat`, ...,
point files, `dims.dat`, `xy.dat`), plus `xbgpu_log.txt` with timings.

GPU-specific keys (optional, in `params.txt`):

| Key | Default | Meaning |
|---|---|---|
| `gpu_solver_acc` | 0.001 | relative residual of the BiCGStab pressure solve (2D) |
| `gpu_maxit` | 200 | maximum solver iterations per step |
| `gpu_hotstart_keepuv` | 1 | keep uu, vv from hotstart files (0 reproduces trunk XBeach, which zeroes them) |

## Supported XBeach options (`wavemodel = nonh`)

- **Grids:** 1D (`ny = 0`) and 2D; `vardx = 0/1` (`xfile`, `yfile`), `alfa`, `gridform = delft3d`
  (`xyfile`), `posdwn`, `xori`/`yori`.
- **Flow:** one layer and reduced two-layer (`nonhq3d`, `nhlay`); `nhbreaker = 0/1` (and 2/3 with
  XBeach-G); `dispc`, `Topt`, `kdmin`; `secorder`, `oldhu`, `umin`, `eps`, `epsi`; viscosity and
  Smagorinsky switches; Coriolis (`lat`, `wearth`); wind (constant or `windfile`).
- **Boundaries:** front `nonh_1d` with `ARC = 0/1`; back `wall`, `abs_1d`, `abs_2d`; laterals `neumann`,
  `wall`, `no_advec`, `neumann_v`, `abs_1d`, or `cyclic = 1`.
- **Waves:** `wbctype = params`, `ts_nonh`, `reuse`, `off`, and spectral boundaries generated on the
  GPU: `parametric` (JONSWAP), `jonstable`, `swan`, `vardens`, with `FILELIST`/`LOCLIST`, `random`,
  bound long waves (`order = 2`) and `highcomp`.
- **Water level:** `zs0`, `zs0file` with `tideloc = 0/1/2/4`, `paulrevere`, `tidetype = velocity/instant`,
  `zsinitfile`; discharges (point and line); rainfall (constant or file).
- **Friction:** `chezy`, `cf`, `manning`, `white-colebrook`, `white-colebrook-grainsize`; `bedfricfile`;
  XBeach-G `friction_acceleration`, `friction_turbulence`, `dynamicroughness`.
- **Processes:** vegetation (bulk drag and the porous canopy model `porcanflow`); groundwater (`gwflow`,
  hydrostatic and non-hydrostatic head, laminar and turbulent, infiltration options); ships (`ships`,
  pressure fields and forces); drifters.
- **Sediment and morphology:** `sedtrans`, `morphology`, all transport formulas XBeach runs in nonh
  (`soulsby_vanrijn`, `vanthiel_vanrijn`, `vanrijn1993`, `nielsen2006`, `mccall_vanrijn`, `intrasedtr`),
  several fractions and bed layers, avalanching, `struct`/`ne_layer`, `morfac` and `morfacopt`,
  `setbathy`, `useXBeachGSettings`.
- **Output:** Fortran and netCDF; global, point, runup-gauge, mean/var/min/max and drifter output;
  `tsglobal`, `tspoints`, `tsmean` time lists; hotstart files read and written (`hotstart`,
  `writehotstart`, `tinth`).

XBeach r6167 accepts the transport formulas `mpm`, `wilcock_crow`, `engelund_fredsoe`, `wong_parker`,
`fl_vb` and `fredsoe_deigaard` but never calls them, so they give no transport; xbnh-gpu does the same.

Not supported, so the program stops with a message: wave-action models (`wavemodel` other than `nonh`),
options that need short-wave forcing (`swrunup`, `tidetype = hybrid`, the quasi-3D sediment profile
`nz > 1`), `gwfastsolve = 1`, `nglobalvar = 999`, and multi-GPU runs.

## Repository

| Path | Content |
|---|---|
| `src/` | the solver: `xbnh_gpu.cu` and its modules (one translation unit) |
| `Makefile`, `install.sh`, `environment.yml` | build and user-level dependencies |
| `xbeach/` | fetches, patches and builds the reference Fortran XBeach r6167 (`environment-xbeach.yml`) |
| `examples/` | deck generators: feature decks, rip-channel beach, cyclic-feature deck |
| `tests/` | quick tests and the feature suite with stored XBeach statistics |
| `tools/` | comparison metrics, benchmark script and plot |
| `docs/` | guide (PDF and LaTeX source) and figures |

## License and credit

XBeach is developed by Deltares, UNESCO-IHE and Delft University of Technology and distributed under the
GNU LGPL. xbnh-gpu is a translation of XBeach routines to CUDA and is distributed under the same license
(LGPL-3.0, see `LICENSE`). The reference XBeach source is not included; `xbeach/build_xbeach.sh`
downloads it from the Deltares SVN. Please cite XBeach (Roelvink et al., 2009,
doi:10.1016/j.coastaleng.2009.08.006) and its non-hydrostatic formulation (Smit et al., 2013,
doi:10.1016/j.coastaleng.2013.01.008) when using this code.
