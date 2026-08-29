"""Run several W-D stage commands CONCURRENTLY inside ONE blocking foreground process.

Why this exists: W-D steps 4-5 have two independent capacity pools - extraction on the Claude
plan path (`claude -p`) and the critic panel on OpenRouter - so critique of an already-extracted
source can overlap extraction of the next one. This driver launches each command as a child
process, streams its stdout to a per-command log, and RETURNS ONLY WHEN EVERY CHILD HAS EXITED.
Nothing is detached: no `&`, no nohup, no orphaned background job.

Usage:
  python wd_parallel.py --log-dir master/logs -- \
      "python extract_fact_sheets.py --source aci --max-seconds 540" \
      "python critique_extracted.py --source primock --route openrouter --max-seconds 540"

Prints a tail of each log plus each exit code, and exits non-zero if any child failed.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    log_dir = os.path.join(HERE, _arg("--log-dir", "master/logs"))
    tail_n = int(_arg("--tail", "25"))
    cmds = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    assert cmds, __doc__
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    procs = []
    for i, cmd in enumerate(cmds):
        path = os.path.join(log_dir, f"{stamp}_{i}_" + "".join(
            ch if ch.isalnum() else "_" for ch in cmd)[:70] + ".log")
        fh = open(path, "w")
        print(f"[start {i}] {cmd}\n         -> {os.path.relpath(path, HERE)}", flush=True)
        procs.append((cmd, path, fh,
                      subprocess.Popen(cmd, shell=True, cwd=HERE, stdout=fh,
                                       stderr=subprocess.STDOUT)))

    t0 = time.time()
    rcs = []
    for cmd, path, fh, p in procs:          # blocking waits, in launch order
        rc = p.wait()
        fh.close()
        rcs.append(rc)
        print(f"\n[done {len(rcs) - 1}] rc={rc} after {time.time() - t0:.0f}s :: {cmd}", flush=True)

    for i, (cmd, path, _fh, _p) in enumerate(procs):
        lines = open(path).read().splitlines()
        print(f"\n=========== tail({tail_n}) of [{i}] {cmd}  (rc={rcs[i]}) ===========")
        print("\n".join(lines[-tail_n:]), flush=True)

    print(f"\nall {len(procs)} commands finished in {time.time() - t0:.0f}s; "
          f"exit codes {rcs}", flush=True)
    sys.exit(1 if any(rcs) else 0)


if __name__ == "__main__":
    main()
