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

if ! phase3_runtime="$(python3 -c 'import platform, numpy; print(platform.python_version() + " " + numpy.__version__)' 2>/dev/null)"; then
  echo "ERROR: activate the locked Phase 3 environment first:" >&2
  echo "  source ~/rtkfree-venvs/phase3/bin/activate" >&2
  exit 2
fi

case "$phase3_runtime" in
  3.12.*\ 2.5.2) ;;
  *)
    echo "ERROR: expected Python 3.12 with NumPy 2.5.2; got $phase3_runtime" >&2
    exit 2
    ;;
esac

python3 -W error -O scripts/healthcheck.py
python3 -W error -m unittest discover -s tests -v
python3 -W error scripts/public_release_check.py
