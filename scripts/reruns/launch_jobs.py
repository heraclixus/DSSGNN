#!/usr/bin/env python3
"""Fan a list of shell commands over the visible GPUs (one job per GPU at a time).

Usage:
  python scripts/reruns/launch_jobs.py JOBFILE --gpus 0,1,2,3,4,5,6,7 --logdir results_local/reruns/<name>

JOBFILE: one job per line, `<logname>\t<command>`; blank lines and lines starting
with # are ignored. Each job runs with CUDA_VISIBLE_DEVICES set to its GPU and
`--device 0` is expected inside the command. Jobs whose log already ends with the
DONE marker are skipped, so the launcher can be re-run to fill in failures.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

DONE = "JOB_DONE_OK"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per GPU")
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for line in open(args.jobfile):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        name, cmd = line.split("\t", 1)
        log = logdir / f"{name}.log"
        if log.exists() and DONE in log.read_text():
            print(f"skip (done): {name}")
            continue
        jobs.append((name, cmd, log))
    slots = [(g, i) for g in gpus for i in range(args.per_gpu)]
    running = {}  # slot -> (Popen, name, log)
    pending = list(jobs)
    print(f"{len(pending)} jobs over {len(slots)} slots", flush=True)
    t0 = time.time()
    while pending or running:
        for slot in slots:
            if slot in running:
                proc, name, log = running[slot]
                if proc.poll() is not None:
                    with open(log, "a") as f:
                        f.write(f"\nEXIT_CODE={proc.returncode}\n" + (DONE + "\n" if proc.returncode == 0 else "JOB_FAILED\n"))
                    print(f"[{time.time()-t0:7.0f}s] finished {name} (exit {proc.returncode})", flush=True)
                    del running[slot]
            if slot not in running and pending:
                name, cmd, log = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0])
                f = open(log, "w")
                f.write(f"# {cmd}\n"); f.flush()
                proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                running[slot] = (proc, name, log)
                print(f"[{time.time()-t0:7.0f}s] started  {name} on gpu {slot[0]}", flush=True)
        time.sleep(5)
    print("ALL_JOBS_FINISHED", flush=True)

if __name__ == "__main__":
    main()
