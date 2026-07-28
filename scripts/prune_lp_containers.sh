#!/usr/bin/env bash
# Remove all lp-rl episode containers (leftovers from interrupted runs).
# Safe: only touches containers named lp-rl-*.
set -uo pipefail
COUNT=$(docker ps -aq --filter name=lp-rl- | wc -l)
docker ps -aq --filter name=lp-rl- | xargs -r docker rm -f >/dev/null
echo "removed ${COUNT} lp-rl containers; remaining: $(docker ps -aq --filter name=lp-rl- | wc -l)"
