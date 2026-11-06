"""Tests for the measurement and report code. No data needed. Run: python tests/test_dentolar.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from dentolar import measurements as dm, report as dr
from dentolar.demo import synthetic

SP = 0.18
I = dm.IDX


def test_known_values():
    m = dm.measure(synthetic(), SP)
    assert abs(m['Root length'] - 400 * SP) < 1e-9 and abs(m['Crown length'] - 300 * SP) < 1e-9
    assert abs(m['Labial bone level'] - 120 * SP) < 1e-9          # crest 120 px apical to CEJA
    assert abs(m['Lingual bone level'] - 70 * SP) < 1e-9
    assert abs(m['Labial bone-level ratio'] - 120 / 400) < 1e-9
    assert abs(m['Labial apical thickness'] - 60 * SP) < 1e-9


def test_rotation_invariance():
    a, b = dm.measure(synthetic(0), SP), dm.measure(synthetic(33), SP)
    for k in dm.MEASUREMENTS:
        assert abs(a[k] - b[k]) < 1e-6, k


def test_bone_level_is_parallel_to_axis_not_perpendicular_to_cej_line():
    # tilt the CEJ line: the perpendicular-to-CEJ-line distance would change, the axis-parallel one must not
    m = dm.measure(synthetic(0, cej_tilt_px=40), SP)
    cejA_y = 400 - 40
    assert abs(m['Labial bone level'] - (520 - cejA_y) * SP) < 1e-9


def test_negative_bone_level_is_kept_and_flagged():
    P = synthetic(); P[I['Acrest']] = (462, 380)                  # crest 20 px coronal to the CEJ
    res = dr.generate_report('T', [P, P], SP)
    row = {r['measurement']: r for r in res['measurements']}
    assert row['Labial bone level']['value'] < 0 and row['Labial bone level']['flag']
    assert row['Labial bone-level ratio']['category'] == 'Low'
    assert res['status'] == 'needs_review'


def test_categories_at_the_boundaries():
    assert dm.categorise_thickness(2.0)[0] == 'Low' and dm.categorise_thickness(1.99)[0] == 'Moderate'
    assert dm.categorise_thickness(1.0)[0] == 'Moderate' and dm.categorise_thickness(0.99)[0] == 'High'
    assert dm.categorise_bone_level_ratio(0.149)[0] == 'Low' and dm.categorise_bone_level_ratio(0.15)[0] == 'Moderate'
    assert dm.categorise_bone_level_ratio(0.33)[0] == 'Moderate' and dm.categorise_bone_level_ratio(0.331)[0] == 'High'
    assert dm.categorise_root_to_crown(2.0)[0] == 'Low' and dm.categorise_root_to_crown(1.5)[0] == 'Moderate'
    assert dm.categorise_root_to_crown(1.0)[0] == 'High'


def test_ensemble_is_mean_of_coordinates():
    P = synthetic(); Q = P + 2.0
    res = dr.generate_report('T', [P, Q], SP)
    assert np.allclose(res['landmarks_ensemble_px'], P + 1.0)


def test_missing_landmark_is_not_reported():
    P = synthetic(); P[3] = np.nan
    assert dr.generate_report('T', [P], SP)['status'] == 'not_reported'


# DICOM reading
def _synthetic_dicom(path, rows=6, cols=8, spacing='0.18\\0.18 ', ts=b'1.2.840.10008.1.2.1\x00'):
    import struct
    def el(g, e, vr, val):
        return struct.pack('<HH', g, e) + vr + struct.pack('<H', len(val)) + val
    ds_el = el(0x0028, 0x0030, b'DS', spacing.encode())                                   # PixelSpacing, NESTED below
    item = struct.pack('<HHI', 0xFFFE, 0xE000, len(ds_el)) + ds_el
    seq = struct.pack('<HH', 0x0028, 0x9110) + b'SQ\x00\x00' + struct.pack('<I', len(item)) + item
    px = (np.arange(rows * cols, dtype='<u2') * 100).tobytes()
    b = b'\x00' * 128 + b'DICM' + el(0x0002, 0x0010, b'UI', ts) + el(0x0028, 0x0010, b'US', struct.pack('<H', rows)) \
        + el(0x0028, 0x0011, b'US', struct.pack('<H', cols)) + el(0x0028, 0x0100, b'US', struct.pack('<H', 16)) \
        + el(0x0028, 0x0103, b'US', struct.pack('<H', 0)) + seq \
        + struct.pack('<HH', 0x7FE0, 0x0010) + b'OW\x00\x00' + struct.pack('<I', len(px)) + px
    open(path, 'wb').write(b)
    return rows, cols


def test_dicom_reader_nested_spacing_and_uint8(tmp_path=None):
    import tempfile, os
    from dentolar import dicom_io
    d = tempfile.mkdtemp(); p = os.path.join(d, 's.dcm'); rows, cols = _synthetic_dicom(p)
    img, sp = dicom_io.read_dicom_slice(p)
    assert img.shape == (rows, cols) and sp == (0.18, 0.18)
    u8 = dicom_io.to_uint8(img)
    assert u8.max() == 255 and u8.dtype == np.uint8
    assert np.array_equal(u8, ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8))  # training rule: min-max
    assert u8.min() == 0
    out = os.path.join(d, 's.png'); sp_mm = dicom_io.dicom_to_png(p, out)
    assert sp_mm == 0.18 and os.path.getsize(out) > 0


def test_compressed_dicom_gives_a_clear_error():
    import tempfile, os
    from dentolar import dicom_io
    d = tempfile.mkdtemp(); p = os.path.join(d, 'c.dcm'); _synthetic_dicom(p, ts=b'1.2.840.10008.1.2.4.70\x00')
    try:
        dicom_io.read_dicom_slice(p)
    except dicom_io.DicomReadError as e:
        assert 'pylibjpeg' in str(e)
    else:
        raise AssertionError('compressed file should raise (when pydicom + decoder are absent)')


def test_pdf_report_is_written():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print('   (matplotlib not installed: PDF test skipped)'); return
    import tempfile, os
    from dentolar.pdf_report import make_pdf
    res = dr.generate_report('PATIENT_CODE_SHOULD_NOT_APPEAR', [synthetic(5)], SP)
    out = os.path.join(tempfile.mkdtemp(), 'r.pdf')
    make_pdf(res, out)
    assert open(out, 'rb').read(4) == b'%PDF' and os.path.getsize(out) > 3000


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1; print('ok  ', name)
    print(n, 'tests passed')
