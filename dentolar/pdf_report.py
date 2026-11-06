"""One-page PDF with the measurement table, review prompts (if any) and the disclaimer.

    make_pdf(result, 'report_ab12cd34.pdf')

No patient information goes into the PDF (tooth_id is not used) and there is no image. Name the file without the
patient code; run.py does that. Needs matplotlib.
"""
import textwrap
from .report import DISCLAIMER, format_row

CAT_COLOR = {'Low': '#2e7d32', 'Moderate': '#9a5b00', 'High': '#b3261e'}


def make_pdf(result, pdf_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as e:
        raise RuntimeError('PDF export needs matplotlib:  pip install matplotlib') from e

    fig = plt.figure(figsize=(8.27, 11.69))                       # A4 portrait
    y = 0.95
    if not result['measurements']:
        fig.text(0.07, y, 'NOT REPORTED', fontsize=13, fontweight='bold', color='#b3261e')
        for f in result.get('flags', []):
            y -= 0.03; fig.text(0.07, y, f, fontsize=9)
    else:
        cols = [0.07, 0.43, 0.60, 0.76]
        for x, h in zip(cols, ['Measurement', 'Value', 'Category', 'Distance to next threshold']):
            fig.text(x, y, h, fontsize=8.5, fontweight='bold')
        fig.add_artist(plt.Line2D([0.07, 0.93], [y - 0.007, y - 0.007], color='#888888', lw=0.8))
        y -= 0.022
        for r in result['measurements']:
            val, dist = format_row(r)
            fig.text(cols[0], y, r['measurement'], fontsize=8.5)
            fig.text(cols[1], y, val, fontsize=8.5)
            if r['category']:
                fig.text(cols[2], y, r['category'], fontsize=8.5, fontweight='bold', color=CAT_COLOR[r['category']])
                fig.text(cols[3], y, dist, fontsize=8.5)
            y -= 0.0215
        flags = result.get('flags') or []
        if flags:
            y -= 0.008
            fig.text(0.07, y, 'Review prompts', fontsize=8.5, fontweight='bold', color='#b3261e'); y -= 0.02
            for f in flags:
                fig.text(0.07, y, '! ' + f.split(' (')[0], fontsize=8, color='#b3261e'); y -= 0.018
        info = f"Ensemble of {result.get('n_models', 1)} fold model(s); pixel spacing {result.get('pixel_spacing_mm')} mm"
        fig.text(0.07, y - 0.008, info, fontsize=7.5, color='#555555'); y -= 0.02
    y -= 0.015
    fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color='#888888', lw=0.8))
    fig.text(0.07, y - 0.01, '\n'.join(textwrap.wrap(DISCLAIMER, 120)), fontsize=7.2, va='top', color='#333333')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return pdf_path
