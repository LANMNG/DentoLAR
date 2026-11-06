"""Interface tests require NumPy only; real checkpoint validation is separate."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import _extract_points, _resolve_files, BEST_EPOCHS, LANDMARKS


class ModelContract(unittest.TestCase):
    def test_original_coordinates_are_not_scaled_clipped_or_averaged(self):
        p = np.arange(32, dtype=float).reshape(1, 16, 2) * 100 - 20
        sample = SimpleNamespace(pred_instances=SimpleNamespace(keypoints=p))
        actual = _extract_points([sample])
        np.testing.assert_array_equal(actual, p[0])
        actual[0, 0] = 99
        self.assertEqual(p[0, 0, 0], -20)

    def test_invalid_prediction_fails(self):
        for p in [np.zeros((1, 15, 2)), np.full((1, 16, 2), np.nan)]:
            with self.assertRaises(RuntimeError):
                _extract_points([SimpleNamespace(pred_instances=SimpleNamespace(keypoints=p))])
        with self.assertRaises(RuntimeError):
            _extract_points([])

    def test_checkpoint_order_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); image = root/'input.png';image.touch()
            base = root/'base.pth';base.touch()
            checkpoints=[]
            for i,e in enumerate(BEST_EPOCHS,1):
                p=root/f'fold_{i}'/f'best_EPE_epoch_{e}.pth';p.parent.mkdir();p.touch();checkpoints.append(p)
                (root/f'fold_{i}.py').touch()
            _,_,actual,_=_resolve_files(image,root,base,None,root)
            self.assertEqual(actual,[p.resolve() for p in checkpoints])
            with self.assertRaises(ValueError):_resolve_files(image,root,base,[checkpoints[0]]*5,root)
            with self.assertRaises(ValueError):_resolve_files(image,root,base,checkpoints[:4],root)
            checkpoints[-1].unlink()
            with self.assertRaises(FileNotFoundError):_resolve_files(image,root,base,None,root)

    def test_measurement_order_matches(self):
        from dentolar.measurements import LANDMARKS as report_order
        self.assertEqual(list(LANDMARKS),report_order)

if __name__ == '__main__':unittest.main()
