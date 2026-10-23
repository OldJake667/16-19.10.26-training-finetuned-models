"""Checks of the attribution math, gradients, hook cleanup, and error selection."""
import unittest
import torch
import numpy as np
from torch import nn
from workflow import gradcam, select_errors, to_spatial


class KnownModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Identity()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[2., -1.], [-3., -2.]]))

    def forward(self, x):
        return self.head(self.spatial(x).mean((2, 3)))


class Tests(unittest.TestCase):
    def test_known_head_and_finite_difference(self):
        model = KnownModel().eval().requires_grad_(False)
        x = torch.tensor([[[[1., 2.], [3., 4.]], [[2., 1.], [4., 3.]]]])
        result = gradcam(model, model.spatial, x)
        expected = (2 * x[0, 0] - x[0, 1]) / 4
        np.testing.assert_allclose(result['maps'][0]['signed_native'], expected.numpy())
        epsilon = 1e-3
        shifted = x.clone()
        shifted[0, 0, 0, 0] += epsilon
        numerical = float((model(shifted)[0, 0] - model(x)[0, 0]) / epsilon)
        self.assertAlmostEqual(numerical, result['maps'][0]['gradient_max_abs'], places=3)
        self.assertTrue(result['maps'][1]['empty_positive_map'])
        self.assertTrue(np.isfinite(result['maps'][1]['normalized_native']).all())
        self.assertEqual(len(model.spatial._forward_hooks), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_tokens_exclude_cls_and_preserve_patch_order(self):
        x = torch.arange(10.).reshape(1, 5, 2)
        output = to_spatial(x, 'tokens', (2, 2))
        np.testing.assert_array_equal(output[0, 0], [[2, 4], [6, 8]])
        with self.assertRaises(ValueError):
            to_spatial(x, 'tokens', (3, 3))

    def test_hook_cleanup_after_failure(self):
        model = KnownModel().eval().requires_grad_(False)
        with self.assertRaises(ValueError):
            gradcam(model, model.spatial, torch.ones(1, 2, 2, 2), 'bad-layout')
        self.assertEqual(len(model.spatial._forward_hooks), 0)

    def test_error_directions_and_no_correct_examples(self):
        errors = select_errors([
            {'file': 'a', 'true_label': 0, 'predicted_label': 1},
            {'file': 'b', 'true_label': 1, 'predicted_label': 0},
            {'file': 'c', 'true_label': 1, 'predicted_label': 1}])
        self.assertEqual([r['file'] for r in errors['true_low_predicted_high']], ['a'])
        self.assertEqual([r['file'] for r in errors['true_high_predicted_low']], ['b'])

    def test_inference_mode_is_rejected(self):
        model = KnownModel().eval().requires_grad_(False)
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            gradcam(model, model.spatial, torch.ones(1, 2, 2, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
