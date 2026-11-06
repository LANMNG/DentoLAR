"""CPU regressions for the previous-zero-loss spike guard."""
import unittest
from types import SimpleNamespace
import torch
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator


class TestZeroLossGuard(unittest.TestCase):
    def run_loss(self, state, value):
        raw = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        loss, logs = TopdownPoseEstimator.parse_losses(state, {"loss_kpt": raw})
        loss.backward()
        self.assertEqual(logs["loss"].item(), value)
        return loss.item(), raw.grad.item()

    def state(self, previous):
        return SimpleNamespace(prev_loss=previous, loss_neighborhood=25.0)

    def test_zero_then_positive_recovers_with_gradient(self):
        state = self.state(None)
        self.assertEqual(self.run_loss(state, 0.0), (0.0, 1.0))
        self.assertEqual(self.run_loss(state, 0.0001), (0.0001, 1.0))
        self.assertEqual(state.prev_loss, 0.0001)

    def test_repeated_zero_then_positive(self):
        state = self.state(0.0)
        for _ in range(3):
            self.assertEqual(self.run_loss(state, 0.0), (0.0, 1.0))
        self.assertEqual(self.run_loss(state, 0.0002), (0.0002, 1.0))

    def test_positive_zero_positive(self):
        state = self.state(0.0001)
        self.assertEqual(self.run_loss(state, 0.0), (0.0, 1.0))
        self.assertEqual(self.run_loss(state, 0.0002), (0.0002, 1.0))

    def test_normal_loss_and_spike_behavior_unchanged(self):
        state = self.state(0.0001)
        self.assertEqual(self.run_loss(state, 0.0002), (0.0002, 1.0))
        self.assertEqual(self.run_loss(state, 0.01), (0.0, 0.0))
        self.assertEqual(state.prev_loss, 0.0002)

    def test_small_nonzero_baseline_retains_spike_detection(self):
        state = self.state(1e-12)
        self.assertEqual(self.run_loss(state, 1e-9), (0.0, 0.0))
        self.assertEqual(state.prev_loss, 1e-12)


if __name__ == "__main__":
    unittest.main()
