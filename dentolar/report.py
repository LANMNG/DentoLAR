"""Turns the landmark coordinates into the measurements and the report.

generate_report(tooth_id, fold_predictions, pixel_spacing_mm)
  fold_predictions: one (16, 2) array per fold model, in pixels of the original exported slice
                    (crop and resize already undone). The five sets are averaged here, before anything is measured.
  pixel_spacing_mm: 0.18 for the export setting used in the study.
It returns a dict with the 13 measurement rows, any review flags, a status ("ok", "needs_review" or
"not_reported") and the report text.
"""
import json
import numpy as np
from . import measurements as dm

DISCLAIMER = ("For research use only. This is not a medical device and not a diagnostic report. Landmarks and "
              "measurements are generated automatically and must be reviewed by a qualified clinician before any use. "
              "The low / moderate / high categories are exploratory thresholds from the literature, not validated risk "
              "categories, and are not a clinical staging or a treatment recommendation.")


def format_row(r):
    """(value text, distance-to-threshold text) of one measurement row."""
    name = r['measurement']; pct = 'bone-level ratio' in name; ratio = 'ratio' in name
    val = f"{r['value'] * 100:.1f} %" if pct else (f"{r['value']:.2f}" if ratio else f"{r['value']:.2f} mm")
    d = r['distance_to_next_threshold']
    dist = '' if d is None else (f'{d * 100:.1f} pp' if pct else (f'{d:.2f}' if ratio else f'{d:.2f} mm'))
    return val, dist


def generate_report(tooth_id, fold_predictions, pixel_spacing_mm=0.18):
    out = dict(tooth_id=tooth_id, pixel_spacing_mm=pixel_spacing_mm)
    try:
        arr = np.asarray(fold_predictions, float)
        arr = arr[None] if arr.ndim == 2 else arr
        if arr.ndim != 3 or arr.shape[1:] != (16, 2) or np.isnan(arr).any():
            raise ValueError(f'need (n_models, 16, 2) coordinates without NaN, got {arr.shape}')
        P = arr.mean(axis=0)
        rows = dm.report(P, pixel_spacing_mm)
    except ValueError as e:                       # missing landmarks, tooth axis parallel to the CEJ line, ...
        out.update(status='not_reported', flags=[f'not reported: {e}'], measurements=[], n_models=0)
        out['text'] = f'DentoLAR report\nNOT REPORTED: {e}'
        return out
    flags = sorted({r['flag'] for r in rows if r['flag']})
    out.update(n_models=int(arr.shape[0]), landmarks_ensemble_px=P.tolist(), measurements=rows, flags=flags,
               status='needs_review' if flags else 'ok')
    out['text'] = format_text(out)
    return out


def format_text(res):
    lines = ['DentoLAR tooth-level report', '-' * 78,
             f"{'Measurement':32s}{'Value':>12s}  {'Category':10s}{'Distance to next threshold':>26s}"]
    for r in res['measurements']:
        val, dist = format_row(r)
        lines.append(f"{r['measurement']:32s}{val:>12s}  {(r['category'] or ''):10s}{dist:>26s}")
        if r['flag']:
            lines.append(f"    ! {r['flag']}")
    return '\n'.join(lines + ['-' * 78, DISCLAIMER])


def report_to_json(res, path):
    open(path, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=2))
