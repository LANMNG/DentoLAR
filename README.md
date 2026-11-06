# DentoLAR

DentoLAR turns a tooth-specific CBCT midsagittal slice of an anterior tooth into a one-page report with 13 dentoalveolar
measurements. A Sapiens model with LoRA finds 16 landmarks on the slice. Everything after that is plain geometry with fixed
rules, so each number in the report can be traced back to the landmarks. For research use only.

```
exported DICOM -> 8-bit PNG -> model.py (landmarks from 5 fold models) -> dentolar/ (measurements, categories) -> PDF + JSON
```

## Quick start
```
pip install -r requirements.txt
python tests/test_dentolar.py
python run.py --demo                              # made-up landmarks, no data or model needed
python run.py --dicom slice.dcm --id CASE_43      # requires the separate model environment and weights
```

## How the measurements are defined
- The tooth axis is the line CP-AP. M is the point where the axis crosses the CEJA-CEJP line.
- Bone level is the distance from the CEJ to the alveolar crest, measured along the tooth axis. It is negative when the
  crest lies coronal to the CEJ, and the report flags that for a second look.
- Bone-level ratio is bone level divided by root length (AP-M). Root-to-crown ratio is AP-M divided by CP-M.
- The low / moderate / high categories use the thresholds in Table 8 of the paper. They are exploratory and are not a
  clinical staging.

## What is in the repository
`dentolar/` has the measurement rules, the report, the PDF and the DICOM reader. `model.py` runs the five fold landmark models, `run.py` runs everything, and `tests/` has the self-tests. Patient data and model weights are not included.

## Local model inference

`model.predict_folds()` is implemented. It returns five separate NumPy arrays,
each `(16, 2)`, in folds 1–5 order. The report layer alone averages them. Coordinate
order matches `dentolar.measurements.LANDMARKS`. Coordinates are in the **original
input image**, because the vendored MMPose estimator already performs the inverse
mapping. The adapter does not crop by an annotation bbox, remap twice or clip points.

Use a dedicated Linux environment with Python 3.10. The tested model environment
uses PyTorch 2.4.0, torchvision 0.19.0 and CUDA 12.1. Install the matching PyTorch
build and CUDA development toolkit first, then run:

```bash
bash scripts/install_model.sh
python -m unittest discover -s tests -p 'test_*.py'
```

The installer builds the vendored MMCV operators for your own environment. No
server binaries are included. The report-only dependencies remain in
`requirements.txt`; a model installation is unnecessary for `--demo`.

### Private weights

Provide both the frozen Sapiens backbone and the five fold checkpoints. The fold
checkpoints contain LoRA parameters and the prediction head, **not the complete
frozen backbone**. They must be used with the same base checkpoint as training.
Obtain trusted weights separately from the project authors. No weights are bundled.

```text
/private/model_weights/
  sapiens_0.3b_epoch_1600_clean.pth
  fold_1/best_EPE_epoch_45.pth
  fold_2/best_EPE_epoch_95.pth
  fold_3/best_EPE_epoch_60.pth
  fold_4/best_EPE_epoch_155.pth
  fold_5/best_EPE_epoch_190.pth
```

```bash
export DENTOLAR_WEIGHTS_DIR=/private/model_weights
export DENTOLAR_DEVICE=cuda:0
python run.py --png /private/slice.png --id CASE01 --out /private/reports
# Or explicit arguments:
python run.py --png /private/slice.png --weights-dir /private/model_weights --device cuda:0
```

Use `DENTOLAR_PRETRAINED` or `--pretrained` to override the base checkpoint location.
The Python API also accepts `checkpoints=[path1, ..., path5]` for arbitrary fold
checkpoint paths and `config_dir=...` for a configuration directory. Paths supplied
by the caller resolve relative to the current working directory. Default config
and default weights directories resolve relative to this repository, so calling
`predict_folds` from another directory works. CPU inference is supported via
`device='cpu'`, but is slow. Fold models load sequentially to limit GPU memory.
Inference does not open SSH connections, read annotations or fetch checkpoints.

### Preserved inference protocol

The five configs under `configs/` share the training architecture: Sapiens-0.3B,
LoRA rank 4 for QKV and rank 8 for projection layers, 16 heatmaps, UDP decoding and
flip-test inference. `ImageContentBBox` detects non-black content using threshold
8, fraction 0.005, minimum 3 pixels, margin 5 and minimum size 16. It then uses
`GetBBoxCenterScale(padding=1.0)` and UDP `TopdownAffine` to preserve aspect ratio
and pad to 768 × 1024. Model normalization matches training. No random training
augmentation is used at inference.

The portable training configs retain per-GPU batch 2, 210 epochs and seed 0. Data
paths in them are placeholders for a separately supplied private dataset. Training
also requires setting `model.backbone.module.init_cfg.checkpoint` to the actual
base checkpoint. The model adapter overrides this automatically for inference.
The full model source, custom crop, zero-loss guard, training/evaluation tools and
original licenses are under `vendor/sapiens/`. Unrelated detection/segmentation
projects, remote queue controllers and experiment outputs are not included.
