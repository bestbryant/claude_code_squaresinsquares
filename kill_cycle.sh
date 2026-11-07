#!/bin/bash
for p in $(pgrep -f "^/bin/bash ./cycle.sh") $(pgrep -f "^/bin/bash ./refresh.sh") $(pgrep -f "tools/getout.py") $(pgrep -f "xargs -P 2"); do kill $p 2>/dev/null; done
pkill -f "^flock -n /tmp/cycle.lock" 2>/dev/null; true
