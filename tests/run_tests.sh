#!/usr/bin/env bash
# Quick tests, a few minutes on any recent GPU:
#   1. rip-channel beach (16k cells, 120 s, ts_nonh boundary, wall back, Neumann laterals): fp64 and fp32,
#   2. Estero-feature deck (40k cells, 100 s: cyclic laterals, abs_2d back, vector U/V/Z boundary, tide
#      file, wind file, point output).
# When the reference XBeach is built (install.sh --with-xbeach), the same decks also run in Fortran XBeach
# (serial for 1, MPI with 2 ranks for 2) and the agreement is checked against thresholds; otherwise the GPU
# results are checked against reference statistics stored in tests/reference.json.
# Usage: bash tests/run_tests.sh      (after: source env.sh)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${WORKDIR:-$ROOT/tests/work}
XS=${XBEACH_SERIAL:-$ROOT/xbeach/dist_serial/bin/xbeach}
XM=${XBEACH_MPI:-$ROOT/xbeach/dist_mpi/bin/xbeach}
rm -rf "$W"; mkdir -p "$W"
pass=0; fail=0
result() { if [ "$1" = 0 ]; then pass=$((pass + 1)); echo "   PASS"; else fail=$((fail + 1)); echo "   FAIL"; fi; }
rungpu() {   # rungpu <deck dir> <run dir> <binary>
  mkdir -p "$2"; cp "$1"/* "$2"/
  (cd "$2" && "$ROOT/bin/$3" > stdout.txt 2>&1); local rc=$?
  echo "   $3: $(tail -1 "$2/stdout.txt")"; return $rc
}

for b in xbnh_gpu xbnh_gpu_sp; do
  [ -x "$ROOT/bin/$b" ] || { echo "bin/$b missing: run install.sh (or make) first"; exit 1; }
done

echo "== 1. rip-channel beach"
python "$ROOT/examples/rip_beach/make_case.py" "$W/rip_deck" --ny 60 --tstop 120 --tintg 1 > /dev/null
rungpu "$W/rip_deck" "$W/rip_gpu" xbnh_gpu; result $?
rungpu "$W/rip_deck" "$W/rip_gpu_sp" xbnh_gpu_sp; result $?
echo " fp32 against fp64"
python "$ROOT/tests/check.py" rip "$W/rip_gpu" "$W/rip_gpu_sp"; result $?
if [ -x "$XS" ]; then
  echo " Fortran XBeach (serial) against GPU"
  mkdir -p "$W/rip_xb"; cp "$W/rip_deck"/* "$W/rip_xb"/
  (cd "$W/rip_xb" && "$XS" > stdout.txt 2>&1)
  python "$ROOT/tests/check.py" rip "$W/rip_xb" "$W/rip_gpu"; result $?
else
  echo " reference statistics (no XBeach build found)"
  python "$ROOT/tests/check.py" ref rip "$W/rip_gpu"; result $?
fi

echo "== 2. Estero-feature deck"
python "$ROOT/examples/estero_features/make_case.py" "$W/est_deck" --tstop 100 > /dev/null
rungpu "$W/est_deck" "$W/est_gpu" xbnh_gpu; result $?
if [ -x "$XM" ] && command -v mpirun > /dev/null; then
  echo " Fortran XBeach (MPI, 2 ranks) against GPU"
  mkdir -p "$W/est_xb"; cp "$W/est_deck"/* "$W/est_xb"/
  (cd "$W/est_xb" && mpirun --oversubscribe -np 2 "$XM" < /dev/null > stdout.txt 2>&1)
  python "$ROOT/tests/check.py" fields "$W/est_xb" "$W/est_gpu"; result $?
else
  echo " reference statistics (no XBeach MPI build found)"
  python "$ROOT/tests/check.py" ref est "$W/est_gpu"; result $?
fi

echo
echo "tests passed: $pass, failed: $fail   (run folders in $W)"
[ $fail = 0 ]
