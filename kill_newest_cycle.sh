#!/bin/bash
p=$(pgrep -n -f "^/bin/bash ./cycle.sh"); [ -n "$p" ] && pkill -TERM -P $p; kill $p 2>/dev/null; echo killed $p
