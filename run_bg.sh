#!/bin/bash
# (re)start background searches; safe to run repeatedly (skips running jobs)
D=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad/hop
mkdir -p $D
cd "$(dirname "$0")/tools"
start() { # name cmd...
  if ! pgrep -f "$2 $3 $4" >/dev/null; then OMP_NUM_THREADS=1 setsid nohup python3 "$@" > $D/$1.log 2>&1 < /dev/null & fi
}
start() { local log=$1; shift; if ! pgrep -f "python3 $*" >/dev/null; then OMP_NUM_THREADS=1 setsid nohup python3 "$@" >> $D/$log.txt 2>&1 < /dev/null & fi; }
start band_0 hopper_band.py 0 2 300 $D
start band_1 hopper_band.py 1 2 301 $D
start hop_0 hopper.py 0 3 100 $D
start hoppq_0 hopper_pq.py 0 1 200 $D
