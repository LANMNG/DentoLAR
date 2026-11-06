#!/usr/bin/env bash
# Run inside a dedicated Python 3.10 / PyTorch environment with a matching CUDA toolkit.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python -c 'import torch, torchvision; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
python -m pip install 'setuptools<81' wheel cython 'numpy==1.26.4'
python -m pip install --no-build-isolation -r requirements-model.txt
python -m pip install -r requirements.txt
for package in engine cv pretrain pose; do
    # The cv package compiles native operators locally; no server binaries are shipped.
    python -m pip install --no-build-isolation --no-deps -e "vendor/sapiens/$package"
done
