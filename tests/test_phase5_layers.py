from __future__ import annotations

import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase5_group import (
    transform_horizontal_vector_channels,
)
from rtkfree_equivariant_gnss_ins.phase5_layers import (
    So2VectorLinear,
    apply_scalar_vector_gate,
    radial_tanh,
)


DTYPE = torch.float64
ATOL = 1e-9
RTOL = 1e-8
ANGLES = torch.tensor((0.37, -1.11, 2.23), dtype=DTYPE)


class Phase5TypedLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(5201)
        self.vectors = torch.randn((3, 4, 2), generator=self.generator, dtype=DTYPE)

    def test_vector_linear_commutes_for_multiple_non_special_angles(self) -> None:
        torch.manual_seed(5202)
        layer = So2VectorLinear(4, 5).to(dtype=DTYPE)
        transformed_then_mapped = layer(
            transform_horizontal_vector_channels(self.vectors, ANGLES)
        )
        mapped_then_transformed = transform_horizontal_vector_channels(
            layer(self.vectors),
            ANGLES,
        )
        torch.testing.assert_close(
            transformed_then_mapped,
            mapped_then_transformed,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_radial_tanh_commutes_and_bounds_norms(self) -> None:
        transformed_then_mapped = radial_tanh(
            transform_horizontal_vector_channels(self.vectors, ANGLES)
        )
        mapped_then_transformed = transform_horizontal_vector_channels(
            radial_tanh(self.vectors),
            ANGLES,
        )
        torch.testing.assert_close(
            transformed_then_mapped,
            mapped_then_transformed,
            atol=ATOL,
            rtol=RTOL,
        )
        self.assertTrue(torch.all(torch.linalg.vector_norm(radial_tanh(self.vectors), dim=-1) < 1.0))
        self.assertTrue(torch.equal(radial_tanh(torch.zeros_like(self.vectors)), torch.zeros_like(self.vectors)))

    def test_invariant_scalar_gate_commutes(self) -> None:
        logits = torch.randn((3, 4), generator=self.generator, dtype=DTYPE)
        transformed_then_gated = apply_scalar_vector_gate(
            transform_horizontal_vector_channels(self.vectors, ANGLES),
            logits,
        )
        gated_then_transformed = transform_horizontal_vector_channels(
            apply_scalar_vector_gate(self.vectors, logits),
            ANGLES,
        )
        torch.testing.assert_close(
            transformed_then_gated,
            gated_then_transformed,
            atol=ATOL,
            rtol=RTOL,
        )

    def test_vector_linear_has_no_non_equivariant_bias(self) -> None:
        layer = So2VectorLinear(4, 5)
        self.assertEqual(sum(parameter.numel() for parameter in layer.parameters()), 40)
        self.assertEqual(set(layer.state_dict()), {"weight_identity", "weight_quarter_turn"})
        self.assertTrue(torch.equal(layer(torch.zeros((4, 2))), torch.zeros((5, 2))))

    def test_shape_and_type_boundaries_are_rejected(self) -> None:
        layer = So2VectorLinear(4, 5)
        with self.assertRaises(ValueError):
            layer(torch.zeros((4, 3)))
        with self.assertRaises(ValueError):
            apply_scalar_vector_gate(torch.zeros((4, 2)), torch.zeros(5))
        with self.assertRaises(TypeError):
            radial_tanh(torch.zeros((4, 2), dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
