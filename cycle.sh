#!/bin/bash
# One maintenance cycle: restart searches, refresh public packings, rebuild + score submission,
# submit to Kaggle if the official score beats the last submitted one, commit + push.
R=$(cd "$(dirname "$0")" && pwd); cd $R
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
LAST=$S/last_submitted_score.txt; [ -f $LAST ] || echo 22.2958508772 > $LAST
true  # hoppers retired (no gains)
./refresh.sh
SC=$(python3 tools/make_submission.py 2>&1 | grep 'official score' | awk '{print $3}')
echo "score $SC (last submitted $(cat $LAST))"
gzip -9 -c submission.csv > submission.csv.gz
if [ -n "$SC" ] && python3 -c "import sys; sys.exit(0 if float('$SC') < float(open('$LAST').read()) - 1e-4 else 1)"; then
  ./submit.sh "auto $SC" && echo $SC > $LAST && echo SUBMITTED
fi
git add -A && git commit -qm "Maintenance cycle: score $SC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JjpoWpE28TFfKCVbwMvSuA"; git push -q origin claude/focused-goldberg-pwu1os
