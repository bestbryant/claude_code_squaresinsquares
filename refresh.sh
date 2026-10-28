#!/bin/bash
# Refresh public notebook outputs, merge into solutions/, run gadget DP, write submission.
set -e
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
R=$(cd "$(dirname "$0")" && pwd)
cd $S
timeout 900 python3 $R/tools/listcode.py > /dev/null 2>&1 || true
timeout 300 python3 $R/tools/frames.py https://www.kaggle.com/code/inversion/squares-in-squares-per-puzzle-leaderboard $S/best_L2.json > /dev/null 2>&1 || true
rm -rf $S/out3 && mkdir -p $S/out3 && cd $S/out3
cat $S/refs.txt | xargs -P 2 -I{} python3 $R/tools/getout.py {} > /dev/null 2>&1 || true
find "$PWD" -name "*.csv" -size +5M | grep -v -E "details|roundtrip" > $S/cands3.txt || true
cd $R
OMP_NUM_THREADS=1 python3 tools/merge.py $(cat $S/cands3.txt) 2>&1 | grep TOTAL
OMP_NUM_THREADS=1 python3 tools/dp_pipeline.py 2>&1 | tail -2
OMP_NUM_THREADS=1 python3 tools/rowcol.py 2>&1 | tail -1
python3 tools/compare.py $S/best_L2.json 3
