#!/bin/bash
# Submit submission.csv to Kaggle. Token is read from ~/.kaggle/access_token (never stored in the repo);
# restored from the session scratchpad after a container restart.
S=/tmp/claude-0/-home-user-claude-code-squaresinsquares/56241591-3644-5216-ad00-1269913e179e/scratchpad
if [ ! -f ~/.kaggle/access_token ] && [ -f $S/.kaggle_token ]; then mkdir -p ~/.kaggle; cp $S/.kaggle_token ~/.kaggle/access_token; chmod 600 ~/.kaggle/access_token; fi
command -v kaggle >/dev/null || pip install -q kaggle
cd "$(dirname "$0")"
kaggle competitions submit -c squares-in-squares -f submission.csv -m "${1:-auto}"
