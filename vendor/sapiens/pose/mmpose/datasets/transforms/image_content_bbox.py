"""Find an image-only ROI before the existing top-down affine transform."""

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImageContentBBox(BaseTransform):
    """Replace annotation boxes with the extent of non-black image content.

    This is black-border removal, not anatomical segmentation. Internal dark
    areas are preserved. Bounds depend ONLY on pixels, never on keypoints or
    annotation boxes. Keep original image coordinates until TopdownAffine so
    the existing target generation and prediction-to-image mapping still work.

    A row/column needs several non-black pixels to reject isolated JPEG noise.
    A small pixel margin protects low-intensity edges. Empty or degenerate
    detections fall back to the whole image. Pixels outside the ROI are set to
    black so TopdownAffine's aspect-ratio expansion provides true letterboxing
    instead of bringing the discarded image margins back into the input.

    Input is the uint8, 0..255 image produced by LoadImage. Use immediately
    after LoadImage and before GetBBoxCenterScale(padding=1.0).
    """

    def __init__(self, threshold=8, min_fraction=0.005, min_pixels=3,
                 margin=5, min_size=16):
        if not 0 <= threshold < 255:
            raise ValueError('threshold must be in [0, 255)')
        if not 0 < min_fraction <= 1:
            raise ValueError('min_fraction must be in (0, 1]')
        if min_pixels < 1 or margin < 0 or min_size < 2:
            raise ValueError('Invalid content-box size parameters')
        self.threshold = threshold
        self.min_fraction = min_fraction
        self.min_pixels = min_pixels
        self.margin = margin
        self.min_size = min_size

    def transform(self, results):
        img = results['img']
        if img.dtype != np.uint8 or img.ndim not in (2, 3):
            raise ValueError('ImageContentBBox expects a uint8 image')
        height, width = img.shape[:2]
        if min(height, width) < 2:
            raise ValueError('Image must be at least 2 x 2 pixels')

        intensity = img.max(axis=2) if img.ndim == 3 else img
        foreground = intensity > self.threshold
        min_row = max(self.min_pixels, int(np.ceil(width * self.min_fraction)))
        min_col = max(self.min_pixels, int(np.ceil(height * self.min_fraction)))
        rows = np.flatnonzero(foreground.sum(axis=1) >= min_row)
        cols = np.flatnonzero(foreground.sum(axis=0) >= min_col)
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1
        if rows.size and cols.size:
            if (cols[-1] - cols[0] + 1 >= self.min_size and
                    rows[-1] - rows[0] + 1 >= self.min_size):
                x1 = max(0, int(cols[0]) - self.margin)
                y1 = max(0, int(rows[0]) - self.margin)
                x2 = min(width - 1, int(cols[-1]) + self.margin)
                y2 = min(height - 1, int(rows[-1]) + self.margin)

        masked = np.zeros_like(img)
        masked[y1:y2 + 1, x1:x2 + 1] = img[y1:y2 + 1, x1:x2 + 1]
        results['img'] = masked
        results['bbox'] = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        results['bbox_score'] = np.ones(1, dtype=np.float32)
        # Do not let previously supplied annotation-derived transforms survive.
        for key in ('bbox_center', 'bbox_scale', 'bbox_rotation'):
            results.pop(key, None)
        return results
