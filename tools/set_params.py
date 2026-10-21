"""
Set keys in an XBeach params.txt in place (adds a key that is missing).

  python tools/set_params.py params.txt tstop=900 writehotstart=1 tinth=900
  python tools/set_params.py params.txt --globalvars zs uu vv     # replace the nglobalvar block

Used by the tests and handy for chaining hotstarted segments.
"""
import re
import sys
from pathlib import Path


def set_key(s, k, v):
    pat = rf"^{re.escape(k)}\s*=.*$"
    if re.search(pat, s, re.M | re.I):
        return re.sub(pat, f"{k} = {v}", s, count=1, flags=re.M | re.I)
    return s.rstrip("\n") + f"\n{k} = {v}\n"


def set_globalvars(s, names):
    # drop "nglobalvar = n" and the n variable names that follow it
    lines, out, skip = s.splitlines(), [], 0
    for line in lines:
        if skip:
            skip -= 1
            continue
        m = re.match(r"^\s*nglobalvar\s*=\s*(\d+)", line, re.I)
        if m:
            skip = int(m.group(1))
            continue
        out.append(line)
    return "\n".join(out) + f"\nnglobalvar = {len(names)}\n" + "".join(n + "\n" for n in names)


def main():
    f = Path(sys.argv[1])
    s = f.read_text()
    args = sys.argv[2:]
    if "--globalvars" in args:
        k = args.index("--globalvars")
        s = set_globalvars(s, args[k + 1:])
        args = args[:k]
    for a in args:
        k, v = a.split("=", 1)
        s = set_key(s, k.strip(), v.strip())
    f.write_text(s)


if __name__ == "__main__":
    main()
