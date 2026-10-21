#!/usr/bin/env bash
# One-time setup on Linux or WSL2, without admin rights:
#   1. micromamba (user-level package manager) in ~/.local/bin, if not already on PATH,
#   2. the build environment (CUDA 12.6 compiler, host compiler, Python) from environment.yml,
#   3. the GPU executables bin/xbnh_gpu (fp64) and bin/xbnh_gpu_sp (fp32),
#   4. with --with-xbeach: the reference Fortran XBeach r6167 (serial and MPI) in xbeach/.
# Usage: ./install.sh [--with-xbeach] [--arch sm_XY]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
WITH_XB=0
ARCH=native
while [ $# -gt 0 ]; do
  case $1 in
    --with-xbeach) WITH_XB=1 ;;
    --arch) ARCH=$2; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

echo "== checking the NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver (on WSL2: the Windows driver) first."; exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader

echo "== micromamba"
if command -v micromamba >/dev/null 2>&1; then
  MM=$(command -v micromamba)
else
  MM=$HOME/.local/bin/micromamba
  if [ ! -x "$MM" ]; then
    mkdir -p "$HOME/.local"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
  fi
fi
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
echo "micromamba: $MM (root $MAMBA_ROOT_PREFIX)"
has_env() { "$MM" env list | awk '{print $1}' | grep -qx "$1"; }

echo "== environment xbnh-gpu"
has_env xbnh-gpu || "$MM" create -y -f "$ROOT/environment.yml"

echo "== building the GPU executables (ARCH=$ARCH)"
"$MM" run -n xbnh-gpu make -C "$ROOT" ARCH="$ARCH"
ls -l "$ROOT/bin"

if [ $WITH_XB = 1 ]; then
  echo "== environment xbnh-xbeach and the reference XBeach build"
  has_env xbnh-xbeach || "$MM" create -y -f "$ROOT/environment-xbeach.yml"
  "$MM" run -n xbnh-xbeach bash "$ROOT/xbeach/build_xbeach.sh"
fi

cat > "$ROOT/env.sh" <<EOF
# source this file to put xbnh-gpu (and the reference XBeach, if built) on PATH
export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX
eval "\$($MM shell hook -s bash)"
micromamba activate xbnh-gpu
export PATH=$ROOT/bin:\$PATH
[ -d /usr/lib/wsl/lib ] && export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}
export XBEACH_SERIAL=$ROOT/xbeach/dist_serial/bin/xbeach
export XBEACH_MPI=$ROOT/xbeach/dist_mpi/bin/xbeach
export XBNH=$ROOT
EOF
echo
echo "Done. Next:  source $ROOT/env.sh   then   bash $ROOT/tests/run_tests.sh"
