#!/usr/bin/env bash
# Quick tests, a few minutes on any recent GPU:
#   1. rip-channel beach (39k cells, 120 s, ts_nonh boundary, wall back, Neumann laterals): fp64 and fp32,
#   2. cyclic-feature deck (40k cells, 100 s: cyclic laterals, abs_2d back, vector U/V/Z boundary, tide
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
python "$ROOT/examples/rip_beach/make_case.py" "$W/rip_deck" --ny 150 --tstop 120 --tintg 1 > /dev/null
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

echo "== 2. cyclic-feature deck"
python "$ROOT/examples/cyclic_features/make_case.py" "$W/cyc_deck" --tstop 100 > /dev/null
rungpu "$W/cyc_deck" "$W/cyc_gpu" xbnh_gpu; result $?
if [ -x "$XM" ] && command -v mpirun > /dev/null; then
  echo " Fortran XBeach (MPI, 2 ranks) against GPU"
  mkdir -p "$W/cyc_xb"; cp "$W/cyc_deck"/* "$W/cyc_xb"/
  (cd "$W/cyc_xb" && mpirun --oversubscribe -np 2 "$XM" < /dev/null > stdout.txt 2>&1)
  python "$ROOT/tests/check.py" fields "$W/cyc_xb" "$W/cyc_gpu"; result $?
else
  echo " reference statistics (no XBeach MPI build found)"
  python "$ROOT/tests/check.py" ref cyc "$W/cyc_gpu"; result $?
fi

echo "== 3. hotstart write and restart (cyclic-feature deck)"
mkdir -p "$W/hs_deck1" "$W/hs_deck2"; cp "$W/cyc_deck"/* "$W/hs_deck1"/
python "$ROOT/tools/set_params.py" "$W/hs_deck1/params.txt" tstop=40 tintg=40 writehotstart=1 tinth=40 --globalvars zs uu vv
rungpu "$W/hs_deck1" "$W/hs_write" xbnh_gpu; result $?
cp "$W/cyc_deck"/* "$W/hs_deck2"/; cp "$W/hs_write"/hotstart_* "$W/hs_deck1/params.txt" "$W/hs_deck2"/
python "$ROOT/tools/set_params.py" "$W/hs_deck2/params.txt" tstop=1 tintg=1 writehotstart=0 hotstart=1 hotstartfileno=1
rungpu "$W/hs_deck2" "$W/hs_read" xbnh_gpu; result $?
python "$ROOT/tests/check.py" hotstart "$W/hs_write" "$W/hs_read"; result $?

echo "== 4. Manning map (bedfricfile): a uniform map must reproduce bedfriccoef"
mkdir -p "$W/fr_deck1" "$W/fr_deck2"; cp "$W/cyc_deck"/* "$W/fr_deck1"/; cp "$W/cyc_deck"/* "$W/fr_deck2"/
python "$ROOT/tools/set_params.py" "$W/fr_deck1/params.txt" tstop=20 tintg=20 writehotstart=0 bedfriction=manning bedfriccoef=0.02 --globalvars zs uu vv
python -c "import numpy as np, sys; z = np.loadtxt(sys.argv[1]); np.savetxt(sys.argv[2], np.full(z.shape, 0.02), fmt='%.3f')" \
  "$W/fr_deck1/z.grd" "$W/fr_deck2/fric.grd"
cp "$W/fr_deck1/params.txt" "$W/fr_deck2/params.txt"
python "$ROOT/tools/set_params.py" "$W/fr_deck2/params.txt" bedfricfile=fric.grd
rungpu "$W/fr_deck1" "$W/fr_coef" xbnh_gpu; result $?
rungpu "$W/fr_deck2" "$W/fr_file" xbnh_gpu; result $?
python "$ROOT/tests/check.py" same "$W/fr_coef" "$W/fr_file"; result $?

echo
echo "tests passed: $pass, failed: $fail   (run folders in $W)"
[ $fail = 0 ]
