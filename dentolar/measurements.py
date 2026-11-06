"""Fixed-rule measurements for one tooth.

Give it the 16 landmarks (pixels) and the pixel spacing (mm). It returns the 13 measurements, the
low / moderate / high categories and how far each value is from the next threshold.

Definitions, same as in the paper:
- tooth axis is the line CP-AP; M is where the axis crosses the CEJA-CEJP line
- thickness is the distance between the two boundary points of a level
- bone level is the distance from the CEJ to the alveolar crest measured along the tooth axis. It keeps its sign:
  positive when the crest is apical to the CEJ, negative when it lies coronal to it
- bone-level ratio is bone level / root length; root length is AP-M, crown length is CP-M,
  root-to-crown ratio is root length / crown length
"""
import numpy as np

# landmark order, same as point_no 1-16 in the annotation and prediction files
LANDMARKS = ['CEJA', 'CEJP', 'AP', 'Acrest', 'Pcrest', 'CP',
             'ABAP', 'PBAP', 'AB1/3', 'AR1/3', 'PR1/3', 'PB1/3',
             'AB1/2', 'AR1/2', 'PR1/2', 'PB1/2']
IDX = {n: i for i, n in enumerate(LANDMARKS)}

MEASUREMENTS = ['Labial apical thickness', 'Labial apical-third thickness', 'Labial midroot thickness',
                'Lingual apical thickness', 'Lingual apical-third thickness', 'Lingual midroot thickness',
                'Labial bone level', 'Lingual bone level', 'Labial bone-level ratio', 'Lingual bone-level ratio',
                'Root length', 'Crown length', 'Root-to-crown ratio']


def _intersect(p1, p2, p3, p4):
    """Intersection of the infinite lines p1-p2 and p3-p4 (None if parallel)."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def measure(landmarks_px, pixel_spacing_mm=0.18):
    """landmarks_px: array (16, 2) of (x, y) in pixels. Returns a dict (values in mm, ratios dimensionless)."""
    P = np.asarray(landmarks_px, float) * pixel_spacing_mm        # to mm
    if P.shape != (16, 2) or np.isnan(P).any():
        raise ValueError('16 landmarks with valid coordinates are required')
    g = lambda name: P[IDX[name]]
    dist = lambda a, b: float(np.linalg.norm(g(a) - g(b)))

    M = _intersect(g('CP'), g('AP'), g('CEJA'), g('CEJP'))        # tooth axis meets CEJ line
    if M is None:
        raise ValueError('tooth axis is parallel to the CEJ line; M cannot be computed')
    axis = g('AP') - g('CP')                                      # unit vector from CP to AP
    axis = axis / np.linalg.norm(axis)

    def bone_level(crest, cej):
        """Signed CEJ-to-crest distance parallel to the tooth axis (positive when the crest is apical to the CEJ)."""
        return float(np.dot(g(crest) - g(cej), axis))

    root = float(np.linalg.norm(g('AP') - M))
    crown = float(np.linalg.norm(g('CP') - M))
    bl_lab = bone_level('Acrest', 'CEJA')
    bl_ling = bone_level('Pcrest', 'CEJP')

    out = {
        'Labial apical thickness':        dist('ABAP', 'AP'),
        'Labial apical-third thickness':  dist('AB1/3', 'AR1/3'),
        'Labial midroot thickness':       dist('AB1/2', 'AR1/2'),
        'Lingual apical thickness':       dist('PBAP', 'AP'),
        'Lingual apical-third thickness': dist('PB1/3', 'PR1/3'),
        'Lingual midroot thickness':      dist('PB1/2', 'PR1/2'),
        'Labial bone level':              bl_lab,
        'Lingual bone level':             bl_ling,
        'Labial bone-level ratio':        bl_lab / root,
        'Lingual bone-level ratio':       bl_ling / root,
        'Root length':                    root,
        'Crown length':                   crown,
        'Root-to-crown ratio':            root / crown,
    }
    out['_M_mm'] = M
    return out


# categories (exploratory, not a risk staging)
# each function returns (category, distance to the next threshold towards the higher category; 0 if already High)
def categorise_thickness(v):                 # low >=2.0, moderate >=1.0 and <2.0, high <1.0 (mm)
    if v >= 2.0: return 'Low', v - 2.0
    if v >= 1.0: return 'Moderate', v - 1.0
    return 'High', 0.0


def categorise_bone_level_ratio(r):          # low <15% (negative ratios count as low), moderate 15-33%, high >33%
    if r < 0.15: return 'Low', 0.15 - r
    if r <= 0.33: return 'Moderate', 0.33 - r
    return 'High', 0.0


def categorise_root_to_crown(r):             # low >=2.0, moderate >1.0 and <2.0, high <=1.0 (Table 8)
    if r >= 2.0: return 'Low', r - 2.0
    if r > 1.0: return 'Moderate', r - 1.0
    return 'High', 0.0


CREST_CORONAL_FLAG = ("Alveolar crest is coronal to the CEJ - please check the landmarks "
                      "(牙槽嵴位于釉牙骨质界冠方，请检查标志点)")


def report(landmarks_px, pixel_spacing_mm=0.18):
    """The 13 measurements, with category and distance to the next threshold for the 9 that have categories.
    A negative bone level (crest coronal to the CEJ) keeps its value and category, so the ratio counts as Low.
    The rows on that side just get a review flag; no number changes."""
    m = measure(landmarks_px, pixel_spacing_mm)
    rows = []
    for name in MEASUREMENTS:
        v = m[name]
        if 'thickness' in name:           cat = categorise_thickness(v)
        elif 'bone-level ratio' in name:  cat = categorise_bone_level_ratio(v)
        elif name == 'Root-to-crown ratio': cat = categorise_root_to_crown(v)
        else:                             cat = (None, None)
        flag = CREST_CORONAL_FLAG if ('bone level' in name or 'bone-level' in name) and \
            ((name.startswith('Labial') and m['Labial bone level'] < 0) or
             (name.startswith('Lingual') and m['Lingual bone level'] < 0)) else None
        rows.append(dict(measurement=name, value=v, category=cat[0], distance_to_next_threshold=cat[1], flag=flag))
    return rows

