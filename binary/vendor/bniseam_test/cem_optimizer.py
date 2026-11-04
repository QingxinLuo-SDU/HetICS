"""Monotone constrained CEM refinement and fixed-label refitting."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .cem_assignment import solve_constrained_assignment
from .cem_objectives import (
    _classification_objective_from_loglik,
    classification_objective,
    dense_center_codes,
    hard_score_matrix,
)
from .cem_types import (
    CEMCandidateResult,
    CEMConfig,
    CandidateSeed,
    RefinementResult,
)
from .feature_transform import FeatureTransform
from .model import (
    FitConfig,
    FitError,
    Parameters,
    StepInfo,
    _hard_label_m_step_with_loglik,
    hard_label_m_step,
    parameter_change,
)
from .multiview_types import ObservedDataView, _require_observed


def _validated_labels(
    labels: np.ndarray,
    n_centers: int,
    k_components: int,
) -> np.ndarray:
    label_array = np.asarray(labels)
    if (
        label_array.ndim != 1
        or label_array.shape != (n_centers,)
        or np.issubdtype(label_array.dtype, np.bool_)
        or not np.issubdtype(label_array.dtype, np.integer)
        or np.any(label_array < 0)
        or np.any(label_array >= k_components)
    ):
        raise ValueError("labels must contain one integer component index per center")
    return label_array.astype(np.int64, copy=True)


def zero_parameters_from_labels(
    data: ObservedDataView,
    labels: np.ndarray,
    k_components: int,
) -> Parameters:
    """Build the frozen zero-regression CEM start from hard labels."""

    observed = _require_observed(data)
    if k_components < 1:
        raise ValueError("k_components must be positive")
    label_array = _validated_labels(
        labels,
        int(observed.centers.size),
        k_components,
    )
    parameters = Parameters(
        pi=np.bincount(label_array, minlength=k_components) / label_array.size,
        gamma=0.0,
        intercepts=np.zeros(k_components),
        beta=np.zeros((k_components, observed.features.shape[1])),
    )
    parameters.validate()
    return parameters


def _step_fields(info: StepInfo) -> dict[str, Any]:
    return {
        "accepted_step_size": info.accepted_step_size,
        "backtracking_count": info.backtracking_count,
        "backtracking_failed": info.backtracking_failed,
        "q_pi_improvement": info.q_pi_new - info.q_pi_old,
        "q_reg_improvement": info.q_reg_new - info.q_reg_old,
    }


def _initial_trace_row(
    objective: float,
    labels: np.ndarray,
    k_components: int,
) -> dict[str, Any]:
    return {
        "phase": "initial",
        "iteration": 0,
        "classification_objective": objective,
        "labels": labels.astype(np.int64, copy=False).tolist(),
        "cluster_sizes": np.bincount(labels, minlength=k_components).tolist(),
        "parameter_change": None,
        "objective_relative_change": None,
        "labels_unchanged": True,
        "accepted": True,
        "accepted_step_size": None,
        "backtracking_count": 0,
        "backtracking_failed": False,
        "q_pi_improvement": None,
        "q_reg_improvement": None,
    }


def _record_solver_event(
    diagnostics: dict[str, Any],
    *,
    phase: str,
    outer_iteration: int,
    status: str,
    detail: dict[str, Any] | None = None,
) -> int:
    events = diagnostics["solver_events"]
    event_bound = diagnostics.get("solver_event_bound")
    if isinstance(event_bound, int) and len(events) >= event_bound:
        raise RuntimeError("predeclared solver-event bound was exceeded")
    event_index = len(events)
    event = {
        "event_index": event_index,
        "phase": phase,
        "outer_iteration": int(outer_iteration),
        "status": status,
    }
    if detail:
        event = {**event, **detail}
    diagnostics["solver_events"] = [*events, event]
    return event_index


def _parameter_digest(parameters: Parameters) -> str:
    digest = hashlib.sha256()
    for value in (
        parameters.pi,
        np.asarray([parameters.gamma], dtype=np.float64),
        parameters.intercepts,
        parameters.beta,
    ):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _record_core_stop(
    diagnostics: dict[str, Any],
    *,
    stopping_time: int,
    status: str,
) -> None:
    diagnostics["stopping_time"] = int(stopping_time)
    diagnostics["core_stop_status"] = status


def _candidate_result(
    seed: CandidateSeed,
    parameters: Parameters | None,
    labels: np.ndarray | None,
    status: str,
    objective: float | None,
    trace: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> CEMCandidateResult:
    eligible = status in {"converged", "max_outer_iter_eligible"}
    return CEMCandidateResult(
        candidate_id=seed.candidate_id,
        method=seed.method,
        parameters=parameters,
        labels=labels,
        status=status,
        eligible=eligible,
        converged=status == "converged",
        training_objective=objective,
        trace=trace,
        diagnostics=diagnostics,
    )


def run_cem_candidate(
    data: ObservedDataView,
    seed: CandidateSeed,
    fit_config: FitConfig,
    cem_config: CEMConfig,
    *,
    audit: Any | None = None,
) -> CEMCandidateResult:
    """Run one finite, monotone, minimum-cluster constrained CEM candidate."""

    observed = _require_observed(data)
    n_centers = int(observed.centers.size)
    center_codes = dense_center_codes(observed)
    k_components = cem_config.k_components
    trace: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "candidate_seed": seed.seed,
        "minimum_cluster_size": None,
        "initial_cluster_sizes": None,
        "final_cluster_sizes": None,
        "outer_iterations": 0,
        "monotonicity_violations": 0,
        "max_objective_decrease": 0.0,
        "stopping_time": None,
        "stopping_time_bound": int(cem_config.max_outer_iter),
        "core_stop_status": None,
        "solver_events": [],
        "solver_event_bound": int(2 * cem_config.max_outer_iter + 2),
        "terminal_pair_event_bound": 2,
        "terminal_refit_iteration_bound": int(fit_config.max_iter),
        "terminal_refit_called": False,
        "terminal_refit_succeeded": False,
        "terminal_assignment_called": False,
        "terminal_assignment_succeeded": False,
        "terminal_same_parameter_verified": False,
    }
    minimum = cem_config.minimum_cluster_size_formula(n_centers)
    diagnostics["minimum_cluster_size"] = minimum
    try:
        cem_config.minimum_cluster_size(n_centers)
        labels = _validated_labels(seed.labels, n_centers, k_components)
    except ValueError as error:
        diagnostics["failure"] = str(error)
        _record_core_stop(diagnostics, stopping_time=0, status="no_feasible_start")
        return _candidate_result(
            seed,
            None,
            None,
            "no_feasible_start",
            None,
            trace,
            diagnostics,
        )

    initial_sizes = np.bincount(labels, minlength=k_components)
    diagnostics["initial_cluster_sizes"] = initial_sizes.tolist()
    if np.any(initial_sizes < minimum):
        diagnostics["failure"] = "initial labels violate minimum cluster size"
        _record_core_stop(diagnostics, stopping_time=0, status="no_feasible_start")
        return _candidate_result(
            seed,
            None,
            labels,
            "no_feasible_start",
            None,
            trace,
            diagnostics,
        )

    try:
        current = zero_parameters_from_labels(observed, labels, k_components)
        objective = classification_objective(observed, current, labels)
    except (FloatingPointError, ValueError) as error:
        diagnostics["failure"] = str(error)
        _record_core_stop(diagnostics, stopping_time=0, status="nonfinite_objective")
        return _candidate_result(
            seed,
            None,
            labels,
            "nonfinite_objective",
            None,
            trace,
            diagnostics,
        )

    trace.append(_initial_trace_row(objective, labels, k_components))
    core_result_status: str | None = None
    for iteration in range(1, cem_config.max_outer_iter + 1):
        old_parameters = current
        old_labels = labels
        old_objective = objective
        try:
            proposed_parameters, info = hard_label_m_step(
                old_parameters,
                old_labels,
                observed.treatment,
                observed.features,
                observed.outcome,
                center_codes,
                fit_config,
            )
            _record_solver_event(
                diagnostics,
                phase="ordinary_refit",
                outer_iteration=iteration,
                status="succeeded",
            )
        except FitError as error:
            diagnostics["failure"] = str(error)
            _record_solver_event(
                diagnostics,
                phase="ordinary_refit",
                outer_iteration=iteration,
                status="failed",
                detail={"failure": str(error)},
            )
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="m_step_backtracking_failed",
            )
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "m_step_backtracking_failed",
                old_objective,
                trace,
                diagnostics,
            )
        except (FloatingPointError, ValueError) as error:
            diagnostics["failure"] = str(error)
            _record_solver_event(
                diagnostics,
                phase="ordinary_refit",
                outer_iteration=iteration,
                status="failed",
                detail={"failure": str(error)},
            )
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="nonfinite_objective",
            )
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "nonfinite_objective",
                old_objective,
                trace,
                diagnostics,
            )

        try:
            scores = hard_score_matrix(observed, proposed_parameters)
        except (FloatingPointError, ValueError) as error:
            diagnostics["failure"] = str(error)
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="nonfinite_objective",
            )
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "nonfinite_objective",
                old_objective,
                trace,
                diagnostics,
            )

        try:
            proposed_labels = solve_constrained_assignment(
                scores,
                min_cluster_size=minimum,
                current_labels=old_labels,
                tolerance=cem_config.assignment_tolerance,
            )
            _record_solver_event(
                diagnostics,
                phase="ordinary_assignment",
                outer_iteration=iteration,
                status="succeeded",
                detail={"assignment_tolerance": cem_config.assignment_tolerance},
            )
        except (RuntimeError, ValueError) as error:
            diagnostics["failure"] = str(error)
            _record_solver_event(
                diagnostics,
                phase="ordinary_assignment",
                outer_iteration=iteration,
                status="failed",
                detail={"failure": str(error)},
            )
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="assignment_infeasible",
            )
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "assignment_infeasible",
                old_objective,
                trace,
                diagnostics,
            )

        try:
            proposed_objective = classification_objective(
                observed,
                proposed_parameters,
                proposed_labels,
            )
            change = parameter_change(old_parameters, proposed_parameters)
            with np.errstate(over="ignore", invalid="ignore"):
                relative_objective_change = float(
                    np.divide(
                        proposed_objective - old_objective,
                        1.0 + abs(old_objective),
                    )
                )
            if not np.isfinite(change) or not np.isfinite(
                relative_objective_change
            ):
                raise FloatingPointError("non-finite CEM stopping metric")
        except (FloatingPointError, ValueError) as error:
            diagnostics["failure"] = str(error)
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="nonfinite_objective",
            )
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "nonfinite_objective",
                old_objective,
                trace,
                diagnostics,
            )

        objective_decrease = max(0.0, old_objective - proposed_objective)
        diagnostics["max_objective_decrease"] = max(
            diagnostics["max_objective_decrease"],
            objective_decrease,
        )
        if objective_decrease > cem_config.objective_tolerance:
            diagnostics["failure"] = (
                "classification objective decreased beyond tolerance"
            )
            diagnostics["monotonicity_violations"] = 1
            diagnostics["rejected_objective"] = proposed_objective
            diagnostics["outer_iterations"] = iteration - 1
            diagnostics["final_cluster_sizes"] = np.bincount(
                old_labels,
                minlength=k_components,
            ).tolist()
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="monotonicity_violation",
            )
            return _candidate_result(
                seed,
                old_parameters,
                old_labels,
                "monotonicity_violation",
                old_objective,
                trace,
                diagnostics,
            )

        labels_unchanged = np.array_equal(proposed_labels, old_labels)
        row = {
            "phase": "ordinary",
            "iteration": iteration,
            "classification_objective": proposed_objective,
            "labels": proposed_labels.astype(np.int64, copy=False).tolist(),
            "cluster_sizes": np.bincount(
                proposed_labels,
                minlength=k_components,
            ).tolist(),
            "parameter_change": change,
            "objective_relative_change": relative_objective_change,
            "labels_unchanged": labels_unchanged,
            "accepted": True,
            **_step_fields(info),
        }
        trace.append(row)
        if audit is not None:
            # Audit-only: write-only sink, never read back by this module.
            audit.record_ordinary(
                iteration=iteration,
                observed=observed,
                parameters=proposed_parameters,
                labels=proposed_labels,
                scores=scores,
                support_size=fit_config.support_size,
                assignment_tolerance=cem_config.assignment_tolerance,
            )
        current = proposed_parameters
        labels = proposed_labels
        objective = proposed_objective
        diagnostics["outer_iterations"] = iteration
        diagnostics["final_cluster_sizes"] = row["cluster_sizes"]

        if (
            labels_unchanged
            and 0.0
            <= relative_objective_change
            <= cem_config.objective_tolerance
            and change <= cem_config.parameter_tolerance
        ):
            core_result_status = "converged"
            _record_core_stop(
                diagnostics,
                stopping_time=iteration,
                status="core_converged",
            )
            break

    if core_result_status is None:
        core_result_status = "max_outer_iter_eligible"
        _record_core_stop(
            diagnostics,
            stopping_time=cem_config.max_outer_iter,
            status="core_max_outer_iter",
        )

    return _terminalize_candidate(
        observed,
        seed,
        current,
        labels,
        objective,
        trace,
        diagnostics,
        fit_config,
        cem_config,
        core_result_status,
        audit=audit,
    )


def refit_fixed_labels(
    data: ObservedDataView,
    initial_parameters: Parameters,
    labels: np.ndarray,
    config: FitConfig,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[Parameters | None, list[dict[str, Any]], str]:
    """Refit all observations with selected labels frozen throughout."""

    observed = _require_observed(data)
    center_codes = dense_center_codes(observed)
    trace: list[dict[str, Any]] = []
    detail = {} if diagnostics is None else diagnostics

    def fail(status: str, reason: str) -> tuple[None, list[dict[str, Any]], str]:
        detail.update(
            {
                "failure_kind": status,
                "failure": reason,
                "converged": False,
            }
        )
        return None, trace, status

    try:
        fixed_labels = _validated_labels(
            labels,
            int(observed.centers.size),
            initial_parameters.k_components,
        )
    except ValueError as error:
        return fail("invalid_input", str(error))

    current = initial_parameters.clone()
    try:
        current.validate()
        objective = classification_objective(observed, current, fixed_labels)
    except FloatingPointError as error:
        return fail("nonfinite_objective", str(error))
    except ValueError as error:
        reason = str(error)
        status = (
            "nonfinite_objective"
            if "NaN" in reason or "infinity" in reason or "non-finite" in reason
            else "invalid_input"
        )
        return fail(status, reason)

    trace.append(_initial_trace_row(objective, fixed_labels, current.k_components))

    # E2 (perf): labels, data and the one-hot responsibilities are fixed for the whole
    # refit, and ``regression_q`` never reads ``pi`` -- so the accepted ``q_reg_new`` of
    # the previous step *is* this step's ``q_reg_old``.  Advanced only once ``proposed``
    # has passed every original check and actually become ``current`` (see below), so a
    # rejected candidate's value can never leak into the next step.
    cached_q_reg_old: float | None = None

    for iteration in range(1, config.max_iter + 1):
        try:
            proposed, info, accepted_loglik = _hard_label_m_step_with_loglik(
                current,
                fixed_labels,
                observed.treatment,
                observed.features,
                observed.outcome,
                center_codes,
                config,
                q_reg_old=cached_q_reg_old,
            )
        except FitError as error:
            return fail("m_step_backtracking_failed", str(error))
        except FloatingPointError as error:
            return fail("nonfinite_objective", str(error))
        except ValueError as error:
            reason = str(error)
            status = (
                "nonfinite_objective"
                if "NaN" in reason or "infinity" in reason or "non-finite" in reason
                else "invalid_input"
            )
            return fail(status, reason)

        try:
            # E3: `proposed` differs from the accepted candidate only in `pi`, which
            # neither the predictor nor the per-row log-likelihood reads -- so the array
            # the M-step just produced is exactly what `classification_objective` would
            # recompute here.  `None` (no candidate accepted) falls back to the original
            # call, which is also what the very first objective above uses.
            if accepted_loglik is None:
                proposed_objective = classification_objective(
                    observed,
                    proposed,
                    fixed_labels,
                )
            else:
                proposed_objective = _classification_objective_from_loglik(
                    observed,
                    proposed,
                    fixed_labels,
                    accepted_loglik,
                    center_codes,
                )
            change = parameter_change(current, proposed)
            with np.errstate(over="ignore", invalid="ignore"):
                relative_objective_change = float(
                    np.divide(
                        proposed_objective - objective,
                        1.0 + abs(objective),
                    )
                )
        except FloatingPointError as error:
            return fail("nonfinite_objective", str(error))
        except ValueError as error:
            return fail("invalid_input", str(error))
        if not np.isfinite(change) or not np.isfinite(relative_objective_change):
            return fail("nonfinite_objective", "non-finite fixed-label stopping metric")
        if proposed_objective + config.q_tolerance < objective:
            return fail(
                "monotonicity_violation",
                "fixed-label classification objective decreased beyond tolerance",
            )

        trace.append(
            {
                "iteration": iteration,
                "classification_objective": proposed_objective,
                "labels": fixed_labels.astype(np.int64, copy=False).tolist(),
                "cluster_sizes": np.bincount(
                    fixed_labels,
                    minlength=current.k_components,
                ).tolist(),
                "parameter_change": change,
                "objective_relative_change": relative_objective_change,
                "labels_unchanged": True,
                "accepted": True,
                **_step_fields(info),
            }
        )
        current = proposed
        objective = proposed_objective
        cached_q_reg_old = info.q_reg_new          # E2: only now, after every check
        if (
            0.0
            <= relative_objective_change
            <= config.objective_relative_tolerance
            and change <= config.parameter_tolerance
        ):
            detail.update(
                {
                    "failure_kind": None,
                    "failure": None,
                    "converged": True,
                    "iterations": iteration,
                }
            )
            return current, trace, "converged"

    detail.update(
        {
            "failure_kind": None,
            "failure": None,
            "converged": False,
            "iterations": config.max_iter,
        }
    )
    return current, trace, "max_iter_eligible"


def _terminalize_candidate(
    observed: ObservedDataView,
    seed: CandidateSeed,
    current: Parameters,
    labels: np.ndarray,
    objective: float,
    trace: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    fit_config: FitConfig,
    cem_config: CEMConfig,
    core_result_status: str,
    audit: Any | None = None,
) -> CEMCandidateResult:
    """Close an ordinary CEM stop with a mandatory same-parameter pair."""

    stopping_time = int(diagnostics["stopping_time"])
    diagnostics["terminal_refit_called"] = True
    refit_diagnostics: dict[str, Any] = {}
    try:
        terminal_parameters, terminal_refit_trace, terminal_refit_status = (
            refit_fixed_labels(
                observed,
                current,
                labels,
                fit_config,
                diagnostics=refit_diagnostics,
            )
        )
    except (FitError, FloatingPointError, RuntimeError, TypeError, ValueError) as error:
        terminal_parameters = None
        terminal_refit_trace = []
        terminal_refit_status = type(error).__name__
        refit_diagnostics.update(
            {"failure_kind": type(error).__name__, "failure": str(error)}
        )
    diagnostics["terminal_refit_status"] = terminal_refit_status
    diagnostics["terminal_refit_trace"] = terminal_refit_trace
    diagnostics["terminal_refit_diagnostics"] = refit_diagnostics
    terminal_refit_max_iter_uncertified = bool(
        terminal_parameters is not None
        and terminal_refit_status == "max_iter_eligible"
    )
    terminal_refit_ok = bool(
        terminal_parameters is not None
        and terminal_refit_status == "converged"
    )
    diagnostics["terminal_refit_succeeded"] = terminal_refit_ok
    diagnostics["terminal_refit_event_index"] = _record_solver_event(
        diagnostics,
        phase="terminal_refit",
        outer_iteration=stopping_time,
        status="succeeded" if terminal_refit_ok else "failed",
        detail={
            "refit_status": terminal_refit_status,
            "internal_iteration_count": max(0, len(terminal_refit_trace) - 1),
        },
    )
    if not terminal_refit_ok or terminal_parameters is None:
        failure_kind = (
            "terminal_refit_max_iter_uncertified"
            if terminal_refit_max_iter_uncertified
            else "terminal_refit_failed"
        )
        failure = (
            "terminal fixed-label refit reached max_iter without a safety certificate"
            if terminal_refit_max_iter_uncertified
            else str(refit_diagnostics.get("failure", terminal_refit_status))
        )
        diagnostics["failure"] = failure
        diagnostics["failure_kind"] = failure_kind
        return _candidate_result(
            seed,
            None,
            labels,
            failure_kind,
            None,
            trace,
            diagnostics,
        )

    terminal_digest = _parameter_digest(terminal_parameters)
    diagnostics["terminal_parameter_digest"] = terminal_digest
    diagnostics["terminal_assignment_called"] = True
    try:
        terminal_scores = hard_score_matrix(observed, terminal_parameters)
        fixed_objective = classification_objective(
            observed,
            terminal_parameters,
            labels,
        )
        terminal_labels = solve_constrained_assignment(
            terminal_scores,
            min_cluster_size=int(diagnostics["minimum_cluster_size"]),
            current_labels=labels,
            tolerance=cem_config.assignment_tolerance,
        )
        assigned_objective = classification_objective(
            observed,
            terminal_parameters,
            terminal_labels,
        )
        if assigned_objective + cem_config.objective_tolerance < fixed_objective:
            raise RuntimeError(
                "terminal reassignment decreased the same-parameter objective"
            )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
        diagnostics["terminal_assignment_succeeded"] = False
        diagnostics["terminal_assignment_event_index"] = _record_solver_event(
            diagnostics,
            phase="terminal_assignment",
            outer_iteration=stopping_time,
            status="failed",
            detail={"failure": str(error)},
        )
        diagnostics["failure"] = str(error)
        diagnostics["failure_kind"] = "terminal_assignment_failed"
        return _candidate_result(
            seed,
            None,
            labels,
            "terminal_assignment_failed",
            None,
            trace,
            diagnostics,
        )

    assignment_digest = _parameter_digest(terminal_parameters)
    diagnostics["terminal_assignment_parameter_digest"] = assignment_digest
    diagnostics["terminal_same_parameter_verified"] = bool(
        terminal_digest == assignment_digest
    )
    diagnostics["terminal_assignment_succeeded"] = True
    diagnostics["terminal_fixed_label_objective"] = float(fixed_objective)
    diagnostics["terminal_assignment_objective"] = float(assigned_objective)
    diagnostics["terminal_assignment_gain"] = float(
        assigned_objective - fixed_objective
    )
    diagnostics["terminal_assignment_regret_upper_bound"] = float(
        cem_config.assignment_tolerance
    )
    if audit is not None:
        # Audit-only: write-only sink, never read back by this module.
        audit.record_terminal(
            scores=terminal_scores,
            labels=terminal_labels,
            assignment_tolerance=cem_config.assignment_tolerance,
        )
    change = parameter_change(current, terminal_parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        relative_change = float(
            np.divide(
                assigned_objective - objective,
                1.0 + abs(objective),
            )
        )
    if not np.isfinite(change) or not np.isfinite(relative_change):
        diagnostics["failure"] = "non-finite terminal stopping metric"
        diagnostics["failure_kind"] = "terminal_assignment_failed"
        diagnostics["terminal_assignment_succeeded"] = False
        diagnostics["terminal_assignment_event_index"] = _record_solver_event(
            diagnostics,
            phase="terminal_assignment",
            outer_iteration=stopping_time,
            status="failed",
            detail={"failure": diagnostics["failure"]},
        )
        return _candidate_result(
            seed,
            None,
            labels,
            "terminal_assignment_failed",
            None,
            trace,
            diagnostics,
        )

    diagnostics["terminal_assignment_event_index"] = _record_solver_event(
        diagnostics,
        phase="terminal_assignment",
        outer_iteration=stopping_time,
        status="succeeded",
        detail={
            "parameter_digest": assignment_digest,
            "assignment_tolerance": cem_config.assignment_tolerance,
            "objective_gain": diagnostics["terminal_assignment_gain"],
            "regret_upper_bound": diagnostics[
                "terminal_assignment_regret_upper_bound"
            ],
        },
    )

    trace.append(
        {
            "phase": "terminal",
            "iteration": stopping_time,
            "classification_objective": float(assigned_objective),
            "labels": terminal_labels.astype(np.int64, copy=False).tolist(),
            "cluster_sizes": np.bincount(
                terminal_labels,
                minlength=cem_config.k_components,
            ).tolist(),
            "parameter_change": float(change),
            "objective_relative_change": relative_change,
            "labels_unchanged": bool(np.array_equal(terminal_labels, labels)),
            "accepted": True,
            "accepted_step_size": None,
            "backtracking_count": 0,
            "backtracking_failed": False,
            "q_pi_improvement": None,
            "q_reg_improvement": None,
            "terminal_refit_status": terminal_refit_status,
            "same_parameter_verified": diagnostics[
                "terminal_same_parameter_verified"
            ],
        }
    )
    diagnostics["final_cluster_sizes"] = trace[-1]["cluster_sizes"]
    terminal_labels_unchanged = bool(np.array_equal(terminal_labels, labels))
    diagnostics["terminal_labels_unchanged"] = terminal_labels_unchanged
    final_status = core_result_status
    if core_result_status == "converged" and (
        terminal_refit_status != "converged" or not terminal_labels_unchanged
    ):
        final_status = "max_outer_iter_eligible"
        diagnostics["strict_convergence_downgraded"] = True
    else:
        diagnostics["strict_convergence_downgraded"] = False
    diagnostics["terminal_output_status"] = final_status
    diagnostics["solver_event_bound_respected"] = bool(
        len(diagnostics["solver_events"]) <= diagnostics["solver_event_bound"]
    )
    return _candidate_result(
        seed,
        terminal_parameters,
        terminal_labels,
        final_status,
        float(assigned_objective),
        trace,
        diagnostics,
    )


def _training_view(
    train: ObservedDataView,
    feature_transform: FeatureTransform,
) -> ObservedDataView:
    """Apply training moments and canonicalize center codes for the kernel."""

    observed = _require_observed(train)
    n_centers = int(observed.centers.size)
    return ObservedDataView(
        features=feature_transform.transform(observed.features),
        treatment=np.asarray(observed.treatment),
        outcome=np.asarray(observed.outcome),
        center_codes=dense_center_codes(observed),
        centers=np.arange(n_centers, dtype=np.int64),
    )


def _result_from_candidate(
    candidate: CEMCandidateResult,
    pre_labels: np.ndarray,
    transform: FeatureTransform,
    minimum: int | None,
) -> RefinementResult:
    post_labels = (
        candidate.labels.copy()
        if candidate.labels is not None
        else pre_labels.copy()
    )
    failure_reason = (
        None if candidate.eligible else str(candidate.diagnostics.get("failure", candidate.status))
    )
    return RefinementResult(
        parameters=candidate.parameters,
        pre_labels=pre_labels.copy(),
        post_labels=post_labels,
        training_objective=candidate.training_objective,
        status=candidate.status,
        eligible=candidate.eligible,
        converged=candidate.converged,
        trace=candidate.trace,
        minimum_cluster_size=minimum,
        failure_reason=failure_reason,
        feature_transform=transform,
        diagnostics=dict(candidate.diagnostics),
    )


def refine_bniseam_candidate(
    train: ObservedDataView,
    initial_labels: np.ndarray,
    *,
    fit_config: FitConfig,
    cem_config: CEMConfig,
    audit: Any | None = None,
) -> RefinementResult:
    """Truth-free fixed-start classification EM refinement."""

    observed = _require_observed(train)
    transform = FeatureTransform.fit(observed.features)
    prepared = _training_view(observed, transform)
    raw_labels = np.asarray(initial_labels)
    pre_labels = raw_labels.copy()
    minimum = cem_config.minimum_cluster_size_formula(int(observed.centers.size))
    candidate = run_cem_candidate(
        prepared,
        CandidateSeed(
            candidate_id=0,
            method="fixed_start",
            labels=raw_labels,
            seed=None,
        ),
        fit_config,
        cem_config,
        audit=audit,
    )
    return _result_from_candidate(candidate, pre_labels, transform, minimum)


def refit_bniseam_fixed_labels(
    train: ObservedDataView,
    labels: np.ndarray,
    *,
    fit_config: FitConfig,
    cem_config: CEMConfig,
    feature_transform: FeatureTransform,
) -> RefinementResult:
    """Truth-free all-training-data refit with center labels held fixed."""

    observed = _require_observed(train)
    transform = feature_transform
    if not isinstance(transform, FeatureTransform):
        raise TypeError("feature_transform must be a FeatureTransform")
    prepared = _training_view(observed, transform)
    raw_labels = np.asarray(labels)
    pre_labels = raw_labels.copy()
    minimum = cem_config.minimum_cluster_size_formula(int(observed.centers.size))
    try:
        fixed_labels = _validated_labels(
            raw_labels,
            int(observed.centers.size),
            cem_config.k_components,
        )
    except ValueError as error:
        return RefinementResult(
            parameters=None,
            pre_labels=pre_labels,
            post_labels=pre_labels.copy(),
            training_objective=None,
            status="invalid_input",
            eligible=False,
            converged=False,
            trace=[],
            minimum_cluster_size=minimum,
            failure_reason=str(error),
            feature_transform=transform,
            diagnostics={"failure_kind": "invalid_input", "failure": str(error)},
        )
    try:
        cem_config.minimum_cluster_size(int(observed.centers.size))
    except ValueError as error:
        return RefinementResult(
            parameters=None,
            pre_labels=pre_labels,
            post_labels=fixed_labels,
            training_objective=None,
            status="no_feasible_start",
            eligible=False,
            converged=False,
            trace=[],
            minimum_cluster_size=minimum,
            failure_reason=str(error),
            feature_transform=transform,
            diagnostics={
                "failure_kind": "no_feasible_start",
                "failure": str(error),
            },
        )
    cluster_sizes = np.bincount(fixed_labels, minlength=cem_config.k_components)
    if np.any(cluster_sizes < minimum):
        failure = "fixed labels violate minimum cluster size"
        return RefinementResult(
            parameters=None,
            pre_labels=pre_labels,
            post_labels=fixed_labels,
            training_objective=None,
            status="no_feasible_start",
            eligible=False,
            converged=False,
            trace=[],
            minimum_cluster_size=minimum,
            failure_reason=failure,
            feature_transform=transform,
            diagnostics={"failure_kind": "no_feasible_start", "failure": failure},
        )

    initial = zero_parameters_from_labels(
        prepared,
        fixed_labels,
        cem_config.k_components,
    )
    refit_diagnostics: dict[str, Any] = {}
    parameters, trace, status = refit_fixed_labels(
        prepared,
        initial,
        fixed_labels,
        fit_config,
        diagnostics=refit_diagnostics,
    )
    eligible = status in {"converged", "max_iter_eligible"} and parameters is not None
    objective = (
        float(trace[-1]["classification_objective"])
        if trace
        else None
    )
    failure_reason = None if eligible else str(
        refit_diagnostics.get("failure", status)
    )
    return RefinementResult(
        parameters=parameters,
        pre_labels=pre_labels,
        post_labels=fixed_labels,
        training_objective=objective,
        status=status,
        eligible=eligible,
        converged=status == "converged",
        trace=trace,
        minimum_cluster_size=minimum,
        failure_reason=failure_reason,
        feature_transform=transform,
        diagnostics={
            **refit_diagnostics,
            "fixed_labels": True,
            "cluster_sizes": cluster_sizes.tolist(),
            "failure": failure_reason,
        },
    )
