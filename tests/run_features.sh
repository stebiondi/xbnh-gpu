#!/usr/bin/env bash
# Feature test suite: one small deck per XBeach option (tests/features/suite.txt), about 5 minutes on the GPU.
# Without --xbeach the GPU results are checked against XBeach statistics stored in tests/features/reference.json.
# With --xbeach the decks also run in Fortran XBeach (serial build from install.sh --with-xbeach, -j decks at
# a time; 20 to 60 minutes) and the global output is compared field by field.
# Usage: bash tests/run_features.sh [--xbeach] [-j N] [--sp] [deck names]   (after: source env.sh)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${WORKDIR:-$ROOT/tests/work_features}
XS=${XBEACH_SERIAL:-$ROOT/xbeach/dist_serial/bin/xbeach}
xb=0; jobs=4; bin=xbnh_gpu; run=gpu; names=()
while [ $# -gt 0 ]; do
  case $1 in
    --xbeach) xb=1 ;;
    -j) jobs=$2; shift ;;
    --sp) bin=xbnh_gpu_sp; run=gpu_sp ;;
    *) names+=("$1") ;;
  esac
  shift
done
[ -x "$ROOT/bin/$bin" ] || { echo "bin/$bin missing: run install.sh (or make) first"; exit 1; }
if [ $xb = 1 ] && [ ! -x "$XS" ]; then echo "no Fortran XBeach at $XS (install.sh --with-xbeach, or set XBEACH_SERIAL)"; exit 1; fi
mkdir -p "$W"
python "$ROOT/tests/features.py" make "$W" "${names[@]}" || exit 1
[ ${#names[@]} -gt 0 ] || read -r -a names <<< "$(python "$ROOT/tests/features.py" list)"

runone() {   # runone <deck name> <run folder> <executable>
  local d="$W/$1"; rm -rf "$d/$2"; mkdir -p "$d/$2"; cp "$d/deck"/* "$d/$2"/
  (cd "$d/$2" && "$3" > stdout.txt 2>&1) || echo "   $1: $2 run failed, see $d/$2/stdout.txt"
}
if [ $xb = 1 ]; then   # XBeach in the background, GPU runs meanwhile
  export -f runone; export W XS
  printf '%s\n' "${names[@]}" | xargs -P "$jobs" -I{} bash -c 'runone {} xb "$XS"' &
  xbpid=$!
fi
n=0
for c in "${names[@]}"; do
  n=$((n + 1)); printf '\r  GPU %d/%d %-14s' $n ${#names[@]} "$c"
  runone "$c" "$run" "$ROOT/bin/$bin"
done
echo
if [ $xb = 1 ]; then echo "  waiting for Fortran XBeach ($jobs at a time)"; wait $xbpid; fi
if [ $xb = 1 ]; then
  python "$ROOT/tests/features.py" compare "$W" --run "$run" "${names[@]}"
else
  python "$ROOT/tests/features.py" ref "$W" --run "$run" "${names[@]}"
fi
