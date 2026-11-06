"""Slice -> landmarks (model.py) -> measurements -> PDF and JSON report.

    python run.py --demo                              # made-up landmarks, no model needed
    python run.py --dicom slice.dcm --id CASE01_43    # whole chain, needs model.py
    python run.py --png slice.png --id CASE01_43      # slice already converted to 8-bit PNG

Everything goes to --out (reports_out by default): report_<random>.pdf, <id>_report.json and report_index.csv.
The PDF name has no patient code. report_index.csv links the id to the PDF, so keep it in the lab.
"""
import argparse, csv, os, uuid
from dentolar import generate_report, report_to_json
from dentolar.pdf_report import make_pdf


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dicom'); g.add_argument('--png'); g.add_argument('--demo', action='store_true')
    ap.add_argument('--id', default='case'); ap.add_argument('--out', default='reports_out')
    ap.add_argument('--spacing', type=float, default=0.18, help='mm per pixel when the DICOM has none (study export: 0.18)')
    ap.add_argument('--weights-dir', help='Directory containing base and five fold checkpoints')
    ap.add_argument('--pretrained', help='Override the frozen Sapiens base checkpoint')
    ap.add_argument('--device', help='For example cuda:0 or cpu; defaults to DENTOLAR_DEVICE or cuda:0')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spacing = a.spacing
    if a.demo:
        from dentolar.demo import synthetic_folds
        folds = synthetic_folds()
    else:
        from model import predict_folds
        png = a.png
        if a.dicom:                                              # step 0: DICOM -> 8-bit PNG (+ spacing from the DICOM)
            from dentolar.dicom_io import dicom_to_png
            png = os.path.join(a.out, f'{a.id}_input.png')
            spacing = dicom_to_png(a.dicom, png, expected_spacing_mm=a.spacing)
        folds = predict_folds(png, weights_dir=a.weights_dir, pretrained=a.pretrained, device=a.device)
    res = generate_report(a.id, folds, spacing)
    report_to_json(res, os.path.join(a.out, f'{a.id}_report.json'))
    pdf = f'report_{uuid.uuid4().hex[:8]}.pdf'
    make_pdf(res, os.path.join(a.out, pdf))
    idx = os.path.join(a.out, 'report_index.csv'); new = not os.path.exists(idx)
    with open(idx, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new: w.writerow(['id', 'pdf_file', 'status'])
        w.writerow([a.id, pdf, res['status']])
    print(res['text']); print(f"\nstatus: {res['status']}   pdf: {os.path.join(a.out, pdf)}")


if __name__ == '__main__':
    main()
