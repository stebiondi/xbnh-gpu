#!/usr/bin/env bash
# GPU (fp64, fp32) versus Fortran XBeach MPI timing on the rip-channel deck, stretched alongshore.
# Same deck, steps and CFL for every code; XBeach time = "MPI timing loop" in XBlog.txt, GPU time = wall
# time of the time loop (xbgpu_log.txt). Output: <workdir>/benchmark.csv; plot with tools/plot_benchmark.py.
# Usage: SIZES="600 2400 9600" RANKS="2 4 8 16" TSTOP=30 bash tools/benchmark.sh [workdir]
# (ny = 33400 gives 8.7 M cells, the size of the Estero Island grid; XBeach MPI needs >= 2 ranks)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${1:-$ROOT/benchmark}
XM=${XBEACH_MPI:-$ROOT/xbeach/dist_mpi/bin/xbeach}
mkdir -p "$W"
CSV=$W/benchmark.csv
[ -f "$CSV" ] || echo "code,ny,cells,ranks,steps,loop_s,s_per_cellstep" > "$CSV"
TSTOP=${TSTOP:-30}
for ny in ${SIZES:-600 2400 9600}; do
  base=$W/ny$ny
  [ -f "$base/boun_U.bcf" ] || python "$ROOT/examples/rip_beach/make_case.py" "$base" --ny "$ny" --tstop "$TSTOP" --noout > /dev/null
  cells=$(( (260 + 1) * (ny + 1) ))
  for b in xbnh_gpu xbnh_gpu_sp; do
    d=$W/ny${ny}_$b; rm -rf "$d"; mkdir "$d"; cp "$base"/{params.txt,z.grd,boun_U.bcf} "$d"/
    (cd "$d" && "$ROOT/bin/$b" > out.txt 2>&1)
    steps=$(awk '/^steps/{print $2}' "$d/xbgpu_log.txt"); wall=$(awk '/^wall_s/{print $2}' "$d/xbgpu_log.txt")
    code=gpu_fp64; [ $b = xbnh_gpu_sp ] && code=gpu_fp32
    echo "$code,$ny,$cells,1,$steps,$wall,$(python -c "print($wall/($steps*$cells))")" >> "$CSV"
    echo "$(date +%T) $code ny=$ny steps=$steps wall=$wall s"
  done
  [ -x "$XM" ] || continue
  for np in ${RANKS:-2 4 8}; do
    d=$W/ny${ny}_mpi$np; rm -rf "$d"; mkdir "$d"; cp "$base"/{params.txt,z.grd,boun_U.bcf} "$d"/
    (cd "$d" && mpirun --oversubscribe --bind-to none -np "$np" "$XM" < /dev/null > out.txt 2>&1)
    steps=$(awk '/Timesteps/{print $3}' "$d/XBlog.txt"); loop=$(awk '/loop  :/{print $3}' "$d/XBlog.txt")
    echo "cpu_mpi,$ny,$cells,$np,$steps,$loop,$(python -c "print($loop/($steps*$cells))")" >> "$CSV"
    echo "$(date +%T) cpu_mpi np=$np ny=$ny steps=$steps loop=$loop s"
    rm -f "$d"/*.dat
  done
done
echo "written $CSV"
