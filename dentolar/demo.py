"""Made-up tooth for the demo and the tests. Not patient data."""
import numpy as np
from . import measurements as dm


def synthetic(angle_deg=0.0, cej_tilt_px=0):
    """Vertical tooth (CP top, AP bottom), horizontal CEJ line (optionally tilted), rotated by angle_deg."""
    I = dm.IDX
    P = np.zeros((16, 2))
    P[I['CP']] = (500, 100); P[I['AP']] = (500, 800)
    P[I['CEJA']] = (470, 400 - cej_tilt_px); P[I['CEJP']] = (530, 400 + cej_tilt_px)
    P[I['Acrest']] = (462, 520); P[I['Pcrest']] = (540, 470)
    for nm, xy in {'ABAP': (440, 800), 'PBAP': (580, 800), 'AB1/3': (455, 645), 'AR1/3': (478, 645),
                   'PR1/3': (522, 645), 'PB1/3': (545, 645), 'AB1/2': (450, 600), 'AR1/2': (475, 600),
                   'PR1/2': (525, 600), 'PB1/2': (552, 600)}.items():
        P[I[nm]] = xy
    a = np.deg2rad(angle_deg); c, s = np.cos(a), np.sin(a)
    return (P - P[I['CP']]) @ np.array([[c, -s], [s, c]]).T + P[I['CP']]


def synthetic_folds(n=5, noise_px=0.6, seed=1):
    """n slightly different copies, standing in for the n fold models."""
    rng = np.random.default_rng(seed)
    P = synthetic(10)
    return [P + rng.normal(0, noise_px, P.shape) for _ in range(n)]
