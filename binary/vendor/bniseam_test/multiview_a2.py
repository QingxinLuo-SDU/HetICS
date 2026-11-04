"""Preregistered finite-envelope A2 calibration for the B2-v3 primary risk."""

from __future__ import annotations

import math
from typing import Any


A2_ACTIVE_SLOT_COUNT = 15
A2_DIRECTION_COUNT = 2
A2_STAGE_COUNT = 2
A2_ENVELOPE_CARD = A2_ACTIVE_SLOT_COUNT * A2_DIRECTION_COUNT * A2_STAGE_COUNT
A2_MIN_VALIDATION_ROWS = 2700
A2_DELTA_Y = 0.025
A2_DELTA_X = 0.025
A2_DELTA_D = A2_DELTA_Y + A2_DELTA_X
A2_CHI = {9: 0.5, 12: 1.0 / 3.0}


def clipped_loss_envelope(epsilon_prob: float) -> float:
    epsilon = float(epsilon_prob)
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 0.5:
        raise ValueError("epsilon_prob must lie in (0, 1/2]")
    return -math.log(epsilon)


def uniform_view_radius(
    *,
    epsilon_prob: float,
    validation_rows: int,
    total_view_budget: float,
    envelope_card: int = A2_ENVELOPE_CARD,
) -> float:
    """Invert the two-sided row-level Hoeffding tail after the record union."""

    if isinstance(validation_rows, bool) or int(validation_rows) != validation_rows or validation_rows < 1:
        raise ValueError("validation_rows must be a positive integer")
    if isinstance(envelope_card, bool) or int(envelope_card) != envelope_card or envelope_card < 1:
        raise ValueError("envelope_card must be a positive integer")
    budget = float(total_view_budget)
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError("total_view_budget must lie in (0, 1)")
    envelope = clipped_loss_envelope(epsilon_prob)
    return envelope * math.sqrt(
        math.log(2.0 * int(envelope_card) / budget) / (2.0 * int(validation_rows))
    )


def preregistered_a2_calibration(*, epsilon_prob: float) -> dict[str, Any]:
    u_y = uniform_view_radius(
        epsilon_prob=epsilon_prob,
        validation_rows=A2_MIN_VALIDATION_ROWS,
        total_view_budget=A2_DELTA_Y,
    )
    u_x = uniform_view_radius(
        epsilon_prob=epsilon_prob,
        validation_rows=A2_MIN_VALIDATION_ROWS,
        total_view_budget=A2_DELTA_X,
    )
    two_u_a2 = u_y + u_x
    return {
        "envelope_card": A2_ENVELOPE_CARD,
        "minimum_validation_rows": A2_MIN_VALIDATION_ROWS,
        "epsilon_prob": float(epsilon_prob),
        "loss_envelope": clipped_loss_envelope(epsilon_prob),
        "delta_y_total_after_union": A2_DELTA_Y,
        "delta_x_total_after_union": A2_DELTA_X,
        "delta_d": A2_DELTA_D,
        "pointwise_tail_y": A2_DELTA_Y / A2_ENVELOPE_CARD,
        "pointwise_tail_x": A2_DELTA_X / A2_ENVELOPE_CARD,
        "u_y": u_y,
        "u_x": u_x,
        "u_a2": (u_y + u_x) / 2.0,
        "two_u_a2": two_u_a2,
        "eta_sel": 0.0,
        "delta_o": 0.0,
        "required_net_gap_e9": two_u_a2 / A2_CHI[9],
        "required_net_gap_e12": two_u_a2 / A2_CHI[12],
        "nonempty_witness_g_min": 1.2,
        "nonempty_witness_defect_over_j_min": 0.0,
    }


__all__ = [
    "A2_ACTIVE_SLOT_COUNT",
    "A2_DIRECTION_COUNT",
    "A2_STAGE_COUNT",
    "A2_ENVELOPE_CARD",
    "A2_MIN_VALIDATION_ROWS",
    "A2_DELTA_Y",
    "A2_DELTA_X",
    "A2_DELTA_D",
    "A2_CHI",
    "clipped_loss_envelope",
    "uniform_view_radius",
    "preregistered_a2_calibration",
]
