#!/bin/bash
# Restart all background search jobs (container restarts kill them).
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
cd /home/user/claude_code_squaresinsquares/tools
pgrep -f "mc_grp.py" >/dev/null || setsid nohup python3 mc_grp.py x 30 3000000 $RANDOM 147 172 199 228 260 292 293 328 329 330 365 366 367 232 >> $S/mcg_1.log 2>&1 &
pgrep -f "gadget_squeeze.py" >/dev/null || (NP=2 setsid nohup sh -c "python3 gadget_squeeze.py $S/gadgets.pkl $S/gdon 240 >> $S/gsq.log 2>&1; python3 dp_pipeline.py $S/gdon > $S/gdp.log 2>&1" >/dev/null 2>&1 &)
cd ..
pgrep -f "cycle.sh" >/dev/null || setsid nohup ./cycle.sh > $S/cycle_last.log 2>&1 &
cd /home/user/claude_code_squaresinsquares/tools
NS="911 910 971 972 498 453 544 803 592 848 973 861 908 791 747 969 591 793 913 994 497 456 634 851 738 490 535 933 641 642 970 593 695 692 975 687 328 914 980 921 932 796 590 632 852 686 996 372 741 703 854 798 790 496 909 492 631 693 976 446"
pgrep -f "sq_run.py" >/dev/null || SQLOG=$S/sq/log_big.txt setsid nohup python3 sq_run.py $S/sq 300 0.12 7 $NS >> $S/sq/log_big.txt 2>&1 &
SPECS=$(python3 -c "print(' '.join(f'{w},{h},{d}' for d in (1,2,3) for w in range(2,8) for h in range(w,9) if w+h<=12 and w*h+d<=60))")
pgrep -f "ls_rect.py" >/dev/null || RESUME=1 setsid nohup python3 ls_rect.py $S/lsrect.pkl 2000000 3 1 $SPECS >> $S/lsrect.log 2>&1 &
