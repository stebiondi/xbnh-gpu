# xbnh-gpu: XBeach non-hydrostatic on the GPU

A CUDA port of the one-layer non-hydrostatic (phase-resolving) flow solver of XBeach (Deltares, trunk
r6167). It reads ordinary XBeach input (`params.txt` and the files it names) and writes XBeach-format
output, so an existing XBeach-NH deck runs unchanged on one NVIDIA GPU.

- Validated against Fortran XBeach: 0.2 % in significant wave height and 0.6 to 0.8 % in mean current
  against serial XBeach. With cyclic laterals it is as close to XBeach MPI as XBeach MPI is to itself
  at a different rank count.
- On a full Estero Island (Florida) storm deck (8.7 M cells, hotstart, cyclic laterals, tide, wind,
  vector wave boundary), an RTX 2000 Ada laptop-class GPU ran 4.1x (fp64) and 9.5x (fp32) faster than
  XBeach on 13 CPU cores.

The full guide, with installation, usage, supported options, validation and performance, is
**[docs/xbnh-gpu_guide.pdf](docs/xbnh-gpu_guide.pdf)**.

## Quick start (Linux or Windows WSL2, no admin rights)

Requirements: an NVIDIA GPU with compute capability 6.0 or newer (Pascal and later). You also need
the NVIDIA driver (on WSL2, the normal Windows driver) and an internet connection for the first install.

```bash
git clone <this repository> xbnh-gpu && cd xbnh-gpu
./install.sh                    # micromamba + CUDA 12.6 toolchain in user space, builds bin/xbnh_gpu{,_sp}
source env.sh                   # puts xbnh_gpu on PATH (do this in every new shell)
bash tests/run_tests.sh         # about 2 minutes
```

To also build the reference Fortran XBeach (serial and MPI, for your own comparisons):

```bash
./install.sh --with-xbeach      # adds gfortran + Open MPI in user space, fetches XBeach r6167, 10-20 min
```

## Running a model

```bash
cd my_xbeach_deck               # an XBeach-NH case: params.txt, bathymetry, boundary files
xbnh_gpu                        # double precision
xbnh_gpu_sp                     # single precision: about 2.3x faster, same statistics in our tests
```

Run the GPU code in its own folder, because its output files have the same names as XBeach's.
Output:
- `zs.dat`, `u.dat`, ... for each `nglobalvar` variable (float64, one (nx+1)(ny+1) record per `tintg`);
- `<name>.dat` for each output point (records [t, variables]);
- `xbgpu_log.txt` with timings.

GPU-specific keys (optional, in `params.txt`):

| Key | Default | Meaning |
|---|---|---|
| `gpu_solver_acc` | 0.001 | relative residual of the BiCGStab pressure solve |
| `gpu_maxit` | 200 | maximum solver iterations per step |
| `gpu_hotstart_keepuv` | 1 | keep uu, vv from hotstart files (0 reproduces trunk XBeach, which zeroes them) |

## Supported XBeach options

`wavemodel = nonh` (one layer, `nonhq3d = 0`) on a uniform rectilinear grid (`vardx = 0`, `alfa = 0`):
- **Offshore boundary:** `front = nonh_1d`, `ARC = 1`, with `wbctype = ts_nonh` (`boun_U.bcf`) or
  `reuse` (`nhbcflist.bcf` with a scalar or vector `nh_series` file of U, V, Z).
- **Back boundary:** `back = wall | abs_1d | abs_2d`.
- **Laterals:** `left`/`right = neumann | wall`, or `cyclic = 1`.
- **Water level, wind, friction:** tide as `zs0` or `zs0file` with `tideloc = 1`; constant wind or
  `windfile`; Manning friction; Smagorinsky viscosity; `nhbreaker = 1`; `secorder = 1`.
- **Restart and output:** ASCII hotstart (`hotstart = 1`), global and point output in Fortran format.

Not supported, so the program stops with a message:
- `nonhq3d = 1`, `vardx = 1`, rotated grids;
- sediment and morphology;
- `tideloc` 2 or 4;
- several `nhbcflist` entries;
- netCDF output, multi-GPU.

## Repository

| Path | Content |
|---|---|
| `src/xbnh_gpu.cu` | the solver (single CUDA file) |
| `Makefile`, `install.sh`, `environment.yml` | build and user-level dependencies |
| `xbeach/` | fetches, patches and builds the reference Fortran XBeach r6167 (`environment-xbeach.yml`) |
| `examples/` | test-deck generators: rip-channel beach, Estero-feature deck |
| `tools/` | comparison metrics, benchmark script and plot |
| `tests/` | quick tests with pass/fail thresholds |
| `docs/` | guide (PDF and LaTeX source) and figures |

## License and credit

XBeach is developed by Deltares, UNESCO-IHE and Delft University of Technology and distributed under the
GNU LGPL. xbnh-gpu is a translation of XBeach routines to CUDA and is distributed under the same license
(LGPL-3.0, see `LICENSE`). The reference XBeach source is not included; `xbeach/build_xbeach.sh`
downloads it from the Deltares SVN. Please cite XBeach (Roelvink et al., 2009, doi:10.1016/j.coastaleng.2009.08.006) and its non-hydrostatic
breaking formulation (Smit et al., 2013, doi:10.1016/j.coastaleng.2013.01.008) when using this code.
