#!/bin/bash
set -euo pipefail
xhost +local:docker >/dev/null 2>&1 || true
docker run --rm -it -v "$PWD":/workspace -w /workspace mqss-dev /bin/bash
