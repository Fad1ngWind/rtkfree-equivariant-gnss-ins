from __future__ import annotations

import inspect
import unittest

import torch

from rtkfree_equivariant_gnss_ins.phase4_state import MeanState
from rtkfree_equivariant_gnss_ins.phase4_student import (
    OrdinaryCausalStudent,
    causal_rollout,
    summarize_imu_interval,
)


DTYPE = torch.float64


def initial_state() -> MeanState:
    return MeanState(
        position_n_m=torch.zeros(3, dtype=DTYPE),
        velocity_n_mps=torch.tensor((1.0, 0.0, 0.0), dtype=DTYPE),
        rotation_n_from_b=torch.eye(3, dtype=DTYPE),
    )


def make_student() -> OrdinaryCausalStudent:
    torch.manual_seed(41)
    return OrdinaryCausalStudent(quality_dim=2, hidden_dim=12).to(dtype=DTYPE)


class Phase4StudentTests(unittest.TestCase):
    def test_imu_summary_is_causal_time_weighted_and_has_no_identity_field(self) -> None:
        acceleration = torch.tensor(((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)), dtype=DTYPE)
        angular_velocity = torch.tensor(((0.1, 0.2, 0.3), (0.3, 0.2, 0.1)), dtype=DTYPE)
        summary = summarize_imu_interval(
            acceleration,
            angular_velocity,
            torch.tensor((0.25, 0.75), dtype=DTYPE),
        )
        expected = torch.tensor((2.5, 2.0, 1.5, 0.25, 0.2, 0.15, 1.0), dtype=DTYPE)
        self.assertTrue(torch.allclose(summary, expected, atol=1e-12, rtol=0.0))

    def test_current_gnss_cannot_change_pre_gnss_prediction(self) -> None:
        student = make_student()
        state = initial_state()
        hidden = student.initial_hidden(state)
        imu = torch.zeros(7, dtype=DTYPE)
        available = torch.ones(1, dtype=DTYPE)
        first = student.forward_step(
            state,
            hidden,
            imu,
            torch.zeros(3, dtype=DTYPE),
            torch.zeros(2, dtype=DTYPE),
            available,
        )
        second = student.forward_step(
            state,
            hidden,
            imu,
            torch.tensor((10.0, -20.0, 5.0), dtype=DTYPE),
            torch.tensor((3.0, -4.0), dtype=DTYPE),
            available,
        )
        self.assertTrue(torch.equal(first.pre_gnss.position_n_m, second.pre_gnss.position_n_m))
        self.assertTrue(torch.equal(first.pre_gnss.velocity_n_mps, second.pre_gnss.velocity_n_mps))
        self.assertTrue(torch.equal(first.pre_gnss.rotation_n_from_b, second.pre_gnss.rotation_n_from_b))
        self.assertFalse(torch.equal(first.posterior.position_n_m, second.posterior.position_n_m))

    def test_masked_gnss_is_fully_hidden_and_posterior_equals_prediction(self) -> None:
        student = make_student()
        state = initial_state()
        hidden = student.initial_hidden(state)
        imu = torch.ones(7, dtype=DTYPE)
        masked = torch.zeros(1, dtype=DTYPE)
        first = student.forward_step(
            state,
            hidden,
            imu,
            torch.zeros(3, dtype=DTYPE),
            torch.zeros(2, dtype=DTYPE),
            masked,
        )
        second = student.forward_step(
            state,
            hidden,
            imu,
            torch.full((3,), 1e6, dtype=DTYPE),
            torch.full((2,), -1e6, dtype=DTYPE),
            masked,
        )
        self.assertTrue(torch.equal(first.gnss_correction, torch.zeros(9, dtype=DTYPE)))
        self.assertTrue(torch.equal(first.posterior.position_n_m, first.pre_gnss.position_n_m))
        self.assertTrue(torch.equal(first.posterior.position_n_m, second.posterior.position_n_m))

    def test_future_perturbation_cannot_change_past_recursive_outputs(self) -> None:
        student = make_student()
        imu = torch.zeros((4, 7), dtype=DTYPE)
        innovation = torch.zeros((4, 3), dtype=DTYPE)
        quality = torch.zeros((4, 2), dtype=DTYPE)
        available = torch.ones((4, 1), dtype=DTYPE)
        baseline = causal_rollout(
            student,
            initial_state(),
            imu,
            innovation,
            quality,
            available,
        )
        changed_imu = imu.clone()
        changed_innovation = innovation.clone()
        changed_imu[3] = 1000.0
        changed_innovation[3] = -1000.0
        perturbed = causal_rollout(
            student,
            initial_state(),
            changed_imu,
            changed_innovation,
            quality,
            available,
        )
        for index in range(3):
            self.assertTrue(
                torch.equal(
                    baseline[index].posterior.position_n_m,
                    perturbed[index].posterior.position_n_m,
                )
            )
        self.assertFalse(
            torch.equal(
                baseline[3].posterior.position_n_m,
                perturbed[3].posterior.position_n_m,
            )
        )

    def test_pre_gnss_api_has_no_teacher_gnss_time_or_identity_argument(self) -> None:
        argument_names = set(inspect.signature(OrdinaryCausalStudent.predict_pre_gnss).parameters)
        prohibited = {"teacher", "covariance", "gnss", "timestamp", "route", "file", "device"}
        self.assertTrue(argument_names.isdisjoint(prohibited))


if __name__ == "__main__":
    unittest.main()
