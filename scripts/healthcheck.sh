#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

case "$repo_root" in
  /mnt/e/rtkfree-equivariant-gnss-ins) ;;
  *)
    echo "ERROR: run only from canonical WSL path /mnt/e/rtkfree-equivariant-gnss-ins" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$repo_root/src"
export PYTHONDONTWRITEBYTECODE=1
python3 -O scripts/healthcheck.py
python3 -m unittest discover -s tests -v
python3 scripts/public_release_check.py
