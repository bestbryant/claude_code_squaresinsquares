#!/bin/bash
# Restart all background search jobs (container restarts kill them).
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
cd /home/user/claude_code_squaresinsquares/tools
pgrep -f "mc_grp.py" >/dev/null || setsid nohup python3 mc_grp.py x 30 3000000 $RANDOM 147 172 199 228 260 292 293 328 329 330 365 366 367 232 >> $S/mcg_1.log 2>&1 &
pgrep -f "gadget_squeeze.py" >/dev/null || (NP=2 setsid nohup sh -c "python3 gadget_squeeze.py $S/gadgets.pkl $S/gdon 240 >> $S/gsq.log 2>&1; python3 dp_pipeline.py $S/gdon > $S/gdp.log 2>&1" >/dev/null 2>&1 &)
cd ..
pgrep -f "cycle.sh" >/dev/null || setsid nohup ./cycle.sh > $S/cycle_last.log 2>&1 &
