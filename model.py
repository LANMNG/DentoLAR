"""Local five-fold Sapiens/LoRA inference in original-image pixel coordinates.

Weights stay outside Git. Set DENTOLAR_WEIGHTS_DIR, DENTOLAR_PRETRAINED and
DENTOLAR_DEVICE, or pass the corresponding keyword arguments. See README.md.
"""
from pathlib import Path
import gc
import os
import numpy as np

ROOT = Path(__file__).resolve().parent
BEST_EPOCHS = (45, 95, 60, 155, 190)
LANDMARKS = ('CEJA', 'CEJP', 'AP', 'Acrest', 'Pcrest', 'CP', 'ABAP', 'PBAP',
             'AB1/3', 'AR1/3', 'PR1/3', 'PB1/3', 'AB1/2', 'AR1/2', 'PR1/2', 'PB1/2')


def _file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}. See README.md for weight layout.')
    return path


def _resolve_files(png_path, weights_dir, pretrained, checkpoints, config_dir):
    image = _file(png_path, 'Input image')
    weights = Path(weights_dir or os.environ.get('DENTOLAR_WEIGHTS_DIR', ROOT / 'weights')).expanduser()
    base = _file(pretrained or os.environ.get('DENTOLAR_PRETRAINED', weights / 'sapiens_0.3b_epoch_1600_clean.pth'), 'Sapiens base checkpoint')
    if checkpoints is None:
        checkpoints = [weights / f'fold_{i}' / f'best_EPE_epoch_{epoch}.pth'
                       for i, epoch in enumerate(BEST_EPOCHS, 1)]
    if len(checkpoints) != 5:
        raise ValueError('Exactly five checkpoints are required, ordered fold 1 through fold 5.')
    ckpts = [_file(p, f'Fold {i} checkpoint') for i, p in enumerate(checkpoints, 1)]
    if len(set(ckpts)) != 5:
        raise ValueError('The five fold checkpoint paths must be distinct.')
    configs = Path(config_dir or ROOT / 'configs').expanduser()
    cfgs = [_file(configs / f'fold_{i}.py', f'Fold {i} config') for i in range(1, 6)]
    return image, base, ckpts, cfgs


def _extract_points(samples):
    if len(samples) != 1:
        raise RuntimeError(f'Expected one tooth image prediction, got {len(samples)}.')
    points = samples[0].pred_instances.keypoints
    if hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (1, 16, 2) or not np.isfinite(points).all():
        raise RuntimeError(f'Invalid landmark prediction: shape={points.shape}; expected finite (1,16,2).')
    # TopdownPoseEstimator.add_pred_to_datasample already maps to the original
    # image. Do NOT apply crop offsets/resize again, clip, reorder or average.
    return points[0].copy()


def predict_folds(png_path, *, weights_dir=None, pretrained=None,
                  checkpoints=None, device=None, config_dir=None):
    """Return five independent (16, 2) arrays, ordered folds 1..5.

    The 16 rows follow LANDMARKS. x/y are pixels in the original input image.
    Loads one fold at a time to avoid holding five models on the GPU. All
    checkpoints must be trusted local files. No annotations or SSH are used.
    """
    image, base, ckpts, cfgs = _resolve_files(
        png_path, weights_dir, pretrained, checkpoints, config_dir)
    # Heavy dependencies are lazy so reporting and --demo work without PyTorch.
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    try:
        import torch
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules
        from mmpose.datasets.transforms.image_content_bbox import ImageContentBBox  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install the vendored Sapiens environment using scripts/install_model.sh; ordinary upstream MMPose is not equivalent.') from exc
    chosen_device = device or os.environ.get('DENTOLAR_DEVICE', 'cuda:0')
    if str(chosen_device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use device="cpu" or DENTOLAR_DEVICE=cpu, or install a CUDA-enabled PyTorch environment.')
    register_all_modules()
    predictions = []
    for checkpoint, config in zip(ckpts, cfgs):
        cfg = Config.fromfile(str(config))
        # LoRAModel initializes the frozen backbone BEFORE loading fold adapters.
        cfg.model.backbone.module.init_cfg = dict(type='Pretrained', checkpoint=str(base))
        model = None
        try:
            model = init_model(cfg, checkpoint=None, device=chosen_device)
            load_checkpoint(model, str(checkpoint), map_location='cpu', strict=True)
            with torch.inference_mode():
                predictions.append(_extract_points(inference_topdown(model, str(image))))
        finally:
            del model
            gc.collect()
            if str(chosen_device).startswith('cuda'):
                torch.cuda.empty_cache()
    return predictions
