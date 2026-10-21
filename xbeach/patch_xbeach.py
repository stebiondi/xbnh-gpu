"""
Two small patches to XBeach trunk r6167, applied by build_xbeach.sh (idempotent, line-ending safe).

1. initialize.F90, flow_init: keep uu and vv read from the hotstart files (trunk zeroes them after
   hotstart_init_1 has read them, so every hotstart restarts from rest). This is the build used for the
   Estero runs, and it matches the xbnh-gpu default (gpu_hotstart_keepuv = 1).
2. postprocess.F90, snappointstogrid: a missing comma in a log-message format ('i0'): ') makes point
   output stop at start-up with "Missing comma between descriptors" (always in serial builds, intermittently
   under MPI). Only the message format changes.
Usage: python patch_xbeach.py <xbeach source dir>
"""
import sys
from pathlib import Path

EDITS = {
    "initialize.F90": (
        "      s%uu=0.d0\n      s%u=0.d0\n      s%vv=0.d0\n",
        "      ! xbnh-gpu patch: keep uu, vv read in hotstart_init_1 (trunk r6167 zeroes them here)\n"
        "      if (par%hotstart .ne. 1) then\n         s%uu=0.d0\n         s%vv=0.d0\n      endif\n"
        "      s%u=0.d0\n"),
    "postprocess.F90": (",i0'): ',", ",i0,'): ',"),
}


def main():
    lib = Path(sys.argv[1]) / "src" / "xbeachlibrary"
    for name, (old, new) in EDITS.items():
        p = lib / name
        raw = p.read_bytes().decode("latin-1")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        if new in text:
            print(f"{name}: already patched")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"{name}: expected text not found once; is this trunk r6167?")
        text = text.replace(old, new)
        p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("latin-1"))
        print(f"{name}: patched")


if __name__ == "__main__":
    main()
