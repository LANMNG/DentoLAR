"""Reads the exported DICOM and gives back an 8-bit PNG plus the pixel spacing.

Two cases:
- Uncompressed DICOM (for example the files converted with ImageJ Bio-Formats, as in the study): works with numpy alone.
- Original CS 3D Imaging export (JPEG lossless): works once  pip install pydicom pylibjpeg pylibjpeg-libjpeg  is in place.
  If that does not work on your machine, convert the files with Bio-Formats first and use the converted ones.

The pixel spacing sits in a nested sequence in the DICOM, not at the top level, so this code searches for it.

The 8-bit rule is the one used to make the training PNGs:
(pixel - min) / (max - min) * 255 per slice, then cast to uint8 (i.e. rounded down). No resizing at this step.
"""
import re, struct, zlib
import numpy as np

UNCOMPRESSED = {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1'}      # implicit / explicit VR little endian
INSTALL_HINT = ('Cannot decode this compressed (JPEG lossless) DICOM. Either  pip install pydicom pylibjpeg '
                'pylibjpeg-libjpeg , or first convert the file to an uncompressed DICOM with ImageJ Bio-Formats '
                '(the route used in the study) and run again.')


class DicomReadError(RuntimeError):
    pass


# small built-in reader for uncompressed files
def _find_ds(h, tag):
    for m in re.finditer(re.escape(tag), h):
        p = m.end()
        if h[p:p + 2] == b'DS':
            ln = struct.unpack('<H', h[p + 2:p + 4])[0]; p += 4
        else:
            ln = struct.unpack('<I', h[p:p + 4])[0]; p += 4
        if 0 < ln < 64:
            txt = h[p:p + ln].decode('latin1').strip('\x00 ')
            if re.fullmatch(r'[0-9eE+\-.\\ ]+', txt):
                return [float(x) for x in txt.replace('\\', ' ').split()]
    return None


def _find_us(h, tag):
    for m in re.finditer(re.escape(tag), h):
        p = m.end()
        if h[p:p + 2] == b'US' and struct.unpack('<H', h[p + 2:p + 4])[0] == 2:
            return struct.unpack('<H', h[p + 4:p + 6])[0]
        if h[p:p + 4] == b'\x02\x00\x00\x00':                       # implicit VR
            return struct.unpack('<H', h[p + 4:p + 6])[0]
    return None


def _read_builtin(path):
    b = open(path, 'rb').read()
    if b[128:132] != b'DICM':
        raise DicomReadError('not a DICOM Part-10 file')
    i = b.find(b'\xe0\x7f\x10\x00', 132)
    if i < 0:
        raise DicomReadError('no pixel data element')
    h = b[:i]
    m = re.search(rb'\x02\x00\x10\x00UI(..)', h, re.S)
    ts = None
    if m:
        ln = struct.unpack('<H', m.group(1))[0]; ts = h[m.end():m.end() + ln].decode().strip('\x00 ')
    if ts not in UNCOMPRESSED:
        raise DicomReadError(INSTALL_HINT + f' (transfer syntax {ts})')
    rows, cols = _find_us(h, b'\x28\x00\x10\x00'), _find_us(h, b'\x28\x00\x11\x00')
    bits, rep = _find_us(h, b'\x28\x00\x00\x01'), _find_us(h, b'\x28\x00\x03\x01')
    if None in (rows, cols, bits) or bits not in (8, 16):
        raise DicomReadError('unsupported or incomplete image attributes')
    p = i + 4
    if b[p:p + 2] in (b'OW', b'OB'):
        n = struct.unpack('<I', b[p + 4:p + 8])[0]; p += 8
    else:
        n = struct.unpack('<I', b[p:p + 4])[0]; p += 4
    dt = np.dtype('<u2' if bits == 16 else 'u1')
    if rep == 1:
        dt = np.dtype('<i2' if bits == 16 else 'i1')
    arr = np.frombuffer(b[p:p + rows * cols * dt.itemsize], dtype=dt).reshape(rows, cols)
    return arr, _find_ds(h, b'\x28\x00\x30\x00'), h


# public functions
def read_dicom_slice(path):
    """Return (image as 2-D numpy array, pixel spacing in mm as (row, col) or None)."""
    try:
        import pydicom
    except ImportError:
        arr, sp, _ = _read_builtin(path)
        return arr, (tuple(sp[:2]) if sp else None)
    ds = pydicom.dcmread(path)
    sp = None
    for el in ds.iterall():                                         # also finds it inside sequences
        if el.tag == (0x0028, 0x0030):
            sp = tuple(float(x) for x in el.value); break
    try:
        arr = ds.pixel_array
    except Exception as e:                                          # for example no JPEG-lossless decoder installed
        raise DicomReadError(f'{e}\n{INSTALL_HINT}')
    if arr.ndim == 3:
        arr = arr[0]
    if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        arr = arr.max() - arr
    return arr, sp


def to_uint8(img):
    """8-bit rule of the training PNGs: per-slice min-max scaling, (pixel - min) / (max - min) * 255, cast to uint8."""
    img = np.asarray(img)
    lo, hi = np.min(img), np.max(img)
    if hi == lo:
        return np.zeros(img.shape, np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)   # same expression as the training script


def save_png(arr8, path):
    """Write an 8-bit grey PNG without any imaging library."""
    a = np.asarray(arr8, np.uint8); h, w = a.shape
    raw = b''.join(b'\x00' + a[r].tobytes() for r in range(h))
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def dicom_to_png(dicom_path, png_path, expected_spacing_mm=0.18, tol=1e-3):
    """DICOM -> 8-bit PNG. Returns the pixel spacing in mm; warns if it differs from the study's 0.18 mm."""
    img, sp = read_dicom_slice(dicom_path)
    save_png(to_uint8(img), png_path)
    if sp is None:
        print(f'WARNING: no PixelSpacing in {dicom_path}; using {expected_spacing_mm} mm')
        return expected_spacing_mm
    if abs(sp[0] - sp[1]) > tol or abs(sp[0] - expected_spacing_mm) > tol:
        print(f'WARNING: pixel spacing {sp} mm differs from the {expected_spacing_mm} mm of the study export; '
              'the model was trained on 0.18 mm slices and the results may not be valid')
    return float(sp[0])
