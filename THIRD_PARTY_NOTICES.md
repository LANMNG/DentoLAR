# Third-party source

`vendor/sapiens/` contains a source-only snapshot of the Sapiens dentistry project,
including its MMPose, MMPretrain, MMCV and MMEngine packages. These are project
forks and should not silently be replaced with current upstream packages.

The supplied upstream root license is **Creative Commons Attribution-NonCommercial
4.0 International**, preserved verbatim at `vendor/sapiens/LICENSE`. Existing
copyright notices remain in individual files. The original Sapiens project is
https://github.com/facebookresearch/sapiens. The source manifest records copied
file hashes and local portability edits. Inclusion here does not relicense
third-party code or weights.

The root `LICENSE` (Apache License 2.0) applies to the DentoLAR code outside
`vendor/` (`dentolar/`, `model.py`, `run.py`, `configs/`, `scripts/` and `tests/`).
Code under `vendor/sapiens/` remains under its own license (CC BY-NC 4.0), so the
landmark localisation model may be used for non-commercial purposes only.
