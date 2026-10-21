#!/usr/bin/env bash
# Reference Fortran XBeach for validating xbnh-gpu: fetch trunk r6167 from the Deltares SVN (anonymous,
# LGPL), apply patch_xbeach.py, and build
#   xbeach/dist_serial/bin/xbeach   serial (no cyclic laterals)
#   xbeach/dist_mpi/bin/xbeach      Open MPI (needed for cyclic = 1; run with mpirun -np N, N >= 2)
# Run inside the xbnh-xbeach environment (install.sh --with-xbeach does this), or with gfortran, Open MPI
# and autotools on PATH. Fortran output only (no netCDF). Takes about 10-20 min (make must be serial).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REV=${XBEACH_REV:-6167}
URL=${XBEACH_URL:-https://svn.oss.deltares.nl/repos/xbeach/trunk}
SRC=$HERE/src_r$REV

if [ ! -d "$SRC" ]; then
  echo "== svn export -r $REV $URL"
  svn export -q --non-interactive -r "$REV" "$URL" "$SRC"
fi
python "$HERE/patch_xbeach.py" "$SRC"

build() {   # build <name> <configure args...>
  local name=$1; shift
  local B=$HERE/build_$name
  if [ -x "$HERE/dist_$name/bin/xbeach" ]; then echo "== $name already built"; return; fi
  echo "== building $name"
  rm -rf "$B"; cp -r "$SRC" "$B"; cd "$B"
  sh autogen.sh > autogen.log 2>&1
  ./configure --prefix="$HERE/dist_$name" --disable-shared "$@" FCFLAGS="-O2" > configure.log 2>&1
  make -j1 > make.log 2>&1 || { tail -30 make.log; exit 1; }
  make install > install.log 2>&1
  echo "   $HERE/dist_$name/bin/xbeach"
}
build serial FC=gfortran
build mpi FC=mpifort --with-mpi
echo "XBeach r$REV ready. Run from a case directory:"
echo "  $HERE/dist_serial/bin/xbeach"
echo "  mpirun -np 4 $HERE/dist_mpi/bin/xbeach"
