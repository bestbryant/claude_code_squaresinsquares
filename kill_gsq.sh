#!/bin/bash
for p in $(pgrep -f "python3 gadget_squeeze"); do kill $p; done
