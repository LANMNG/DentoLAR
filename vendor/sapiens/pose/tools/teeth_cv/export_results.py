"""Export raw-image coordinates and legacy-compatible teeth measurements."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

NAMES = ['CEJA', 'CEJP', 'AP', 'A_crest', 'P_crest', 'CP', 'AB_AP', 'PB_AP',
         'AB_1/3', 'AR_1/3', 'PR_1/3', 'PB_1/3', 'AB_1/2', 'AR_1/2',
         'PR_1/2', 'PB_1/2']
CSV_NAMES = ['CEJ_A', 'CEJ_P'] + NAMES[2:]
HEADER = ['image', 'set', 'fold', 'model', 'point_no', 'landmark', 'x_px', 'y_px']
SPACING = 1 / 5.5556  # identical to the existing project evaluation script


def measurements(p):
    """Same 13 values/order as Measurement_GT/Measurement_pre in evaluation/.

    Degenerate/non-intersecting segments yield NaN for undefined quantities,
    and are explicitly reported, rather than crashing or fabricating values.
    """
    values = [float(np.linalg.norm(p[a] - p[b]) * SPACING)
              for a, b in [(6, 2), (8, 9), (12, 13), (7, 2), (11, 10), (15, 14)]]
    issues = []
    cej = p[1] - p[0]
    norm = np.linalg.norm(cej)
    if norm <= 1e-10:
        da = dp = float('nan')
        issues.append('coincident_CEJ_points')
    else:
        da = float(abs(np.cross(cej, p[3] - p[0])) / norm * SPACING)
        dp = float(abs(np.cross(cej, p[4] - p[0])) / norm * SPACING)
    intersection = LineString([p[2], p[5]]).intersection(LineString([p[0], p[1]]))
    if intersection.is_empty or intersection.geom_type != 'Point':
        ap = cp = float('nan')
        issues.append('AP_CP_and_CEJ_segments_do_not_intersect_at_one_point')
    else:
        m = np.array([intersection.x, intersection.y])
        ap = float(np.linalg.norm(p[2] - m) * SPACING)
        cp = float(np.linalg.norm(p[5] - m) * SPACING)
    def ratio(a, b):
        if not np.isfinite(b) or b <= 1e-10:
            issues.append('undefined_ratio_denominator')
            return float('nan')
        return float(a / b)
    values += [da, dp, ratio(da, ap), ratio(dp, ap), ap, cp, ratio(ap, cp)]
    return values, sorted(set(issues))


def export(annotations, predictions, output, fold, split, template):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(template, newline='') as f:
        if next(csv.reader(f)) != HEADER:
            raise ValueError('Prediction template columns have changed')
    data = json.loads(Path(annotations).read_text())
    with open(predictions, 'rb') as f:
        dumped = pickle.load(f)  # own/local trusted model output only
    pred = {}
    for sample in dumped:
        name = Path(sample['img_path']).name
        if name in pred:
            raise ValueError(f'Duplicate prediction: {name}')
        points = np.asarray(sample['pred_instances']['keypoints'], dtype=float)
        if points.shape != (1, 16, 2) or not np.isfinite(points).all():
            raise ValueError(f'Invalid prediction shape/values: {name} {points.shape}')
        pred[name] = points[0]
    images = {im['id']: im['file_name'] for im in data['images']}
    gt = {}
    for ann in data['annotations']:
        name = images[ann['image_id']]
        if name in gt:
            raise ValueError(f'Multiple annotations per image: {name}')
        points = np.asarray(ann['keypoints'], dtype=float).reshape(16, 3)
        if not np.isfinite(points).all() or not np.all(points[:, 2] > 0):
            raise ValueError('Legacy metrics require all 16 landmarks to be visible')
        gt[name] = points[:, :2]
    if set(gt) != set(pred) or len(gt) != len(images):
        raise ValueError(f'Prediction/annotation coverage mismatch: {len(gt)}, {len(pred)}')
    names = sorted(gt)
    with (output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(HEADER)
        for name in names:
            for i, (x, y) in enumerate(pred[name]):
                writer.writerow([name, split, fold if split == 'val' else '',
                                 f'fold{fold}', i + 1, CSV_NAMES[i], f'{x:.6f}', f'{y:.6f}'])
    errors_px = np.stack([np.linalg.norm(pred[n] - gt[n], axis=1) for n in names])
    errors = errors_px * SPACING
    means, stds = errors.mean(axis=0), errors.std(axis=0)
    with (output / 'point_results.txt').open('w') as f:
        f.write('Means of each point:\n')
        for name, value in zip(NAMES, means): f.write(f'{name}: {value}\n')
        f.write('\nStd of each point:\n')
        for name, value in zip(NAMES, stds): f.write(f'{name}: {value}\n')
        f.write(f'\nmean_all: {means.mean()}\nStd_all: {stds.mean()}\n')
    issues = []
    with (output / 'measurement_details.txt').open('w') as f:
        for title, source in [('GT', gt), ('Prediction', pred)]:
            f.write(f'Measurements of {title}:\n')
            for name in names:
                values, problems = measurements(source[name])
                f.write(f'{name}: {values}\n')
                if problems: issues.append(dict(image=name, source=title, issues=problems))
            f.write('\n')
    summary = dict(fold=fold, split=split, images=len(names), csv_rows=len(names)*16,
                   EPE_px=float(errors_px.mean()), point_error_scaled=float(means.mean()),
                   spacing=SPACING, spacing_source='existing project evaluation, not independently calibrated',
                   Std_all_definition='mean of 16 population standard deviations (legacy)',
                   measurement_issues=issues)
    (output / 'export_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['annotations', 'predictions', 'output', 'template']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    a = p.parse_args()
    print(json.dumps(export(a.annotations, a.predictions, a.output, a.fold,
                            a.split, a.template), indent=2))
