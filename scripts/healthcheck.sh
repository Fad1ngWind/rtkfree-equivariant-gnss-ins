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

if ! phase4_runtime="$(python3 -c 'import platform, numpy, torch; print(platform.python_version() + " " + numpy.__version__ + " " + torch.__version__ + " " + str(torch.version.cuda))' 2>/dev/null)"; then
  echo "ERROR: activate the locked Phase 4 environment first:" >&2
  echo "  source ~/rtkfree-venvs/phase4/bin/activate" >&2
  exit 2
fi

case "$phase4_runtime" in
  3.12.*\ 2.5.2\ 2.13.0+cpu\ None) ;;
  *)
    echo "ERROR: expected Python 3.12, NumPy 2.5.2, and CPU-only PyTorch 2.13.0; got $phase4_runtime" >&2
    exit 2
    ;;
esac

python3 -W error -O scripts/healthcheck.py
python3 -W error -m unittest discover -s tests -v
python3 -W error scripts/public_release_check.py
