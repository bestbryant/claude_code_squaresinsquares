#!/bin/bash
# One maintenance cycle: restart searches, refresh public packings, rebuild + score submission,
# submit to Kaggle if the official score beats the last submitted one, commit + push.
R=$(cd "$(dirname "$0")" && pwd); cd $R
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
mkdir -p $S
# last submitted score lives in the repo so it survives container resets
LAST=$R/.last_submitted_score; [ -f $LAST ] || echo 22.106224887301273 > $LAST
true  # hoppers retired (no gains)
./refresh.sh
CH=$(git diff --name-only -- solutions; git ls-files --others --exclude-standard solutions); [ -n "$CH" ] && python3 tools/robustify.py $CH 2>&1 | grep -v Warn | tail -1
SC=$(python3 tools/make_submission.py 2>&1 | grep 'official score' | awk '{print $3}')
echo "score $SC (last submitted $(cat $LAST))"
# never commit a broken state: stop if the submission could not be built/scored
if [ -z "$SC" ]; then echo "ABORT: no official score (make_submission failed)"; exit 1; fi
gzip -9 -c submission.csv > submission.csv.gz
if [ -n "$SC" ] && python3 -c "import sys; sys.exit(0 if float('$SC') < float(open('$LAST').read()) - 1e-4 else 1)"; then
  ./submit.sh "auto $SC" && echo $SC > $LAST && echo SUBMITTED
fi
git add -A && git commit -qm "Maintenance cycle: score $SC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JjpoWpE28TFfKCVbwMvSuA"; git push -q origin claude/focused-goldberg-pwu1os
