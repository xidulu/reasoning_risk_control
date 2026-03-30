import numpy as np
import json
from signal_util import ema_causal

def sigmoid_func(tokens, shift=0, c=0.1, upper_bound=1.0, lower=0.0):
    x = (tokens - shift) * c
    val = 1 / (1 + np.exp(-x))
    return val * (upper_bound - lower) + lower


def threshold_to_loss_vectorized_upper(
        entry,
        thresholds,
        exit_time_fn,
        signal_field='confidence',
        correctness_field='intermediate_correct',
        signal_transform_fn=ema_causal,
    ):
    assert len(thresholds.shape) == 1
    N = len(thresholds)
    signal_values = entry[signal_field]
    correctness = entry[correctness_field]
    if signal_transform_fn is not None:
        signal_values = signal_transform_fn(signal_values)
    exit_time = exit_time_fn(signal_values, thresholds)  # (N,)
    exit_time = np.clip(exit_time, 0, correctness.size - 1)
    corr_at_exit = correctness[exit_time]
    correctness_loss = (~corr_at_exit).astype(np.float32)
    if correctness.any():
        first_correct_exit = int(np.argmax(correctness))  # first True
        regret = np.maximum(0, exit_time - first_correct_exit).astype(np.float32)
        efficiency_loss = regret 
        # / float(correctness.size)
    else:
        efficiency_loss = np.zeros(N, dtype=np.float32)
    return correctness_loss, efficiency_loss


def lower_threshold_correctness_risk_v1(correctness, exit_points):
    """
    correctness: (T,) bool/int, per-step correctness, final correctness = correctness[-1]
    exit_points: (M,) int, exit time index in [0, T-1], or `no_exit` for no exit

    returns:
      risks_per_threshold: (M,) {0,1}
    """
    correctness = np.asarray(correctness).astype(bool)
    exit_points = np.asarray(exit_points)

    T = correctness.shape[0]
    final_correct = correctness[-1]
    no_exit = len(correctness) - 1
    # "no exit => no risk"
    did_exit = (exit_points != no_exit)

    # "exited early" (exiting at the final step is effectively not truncating)
    exited_early = did_exit & (exit_points < T - 1)

    return (final_correct & exited_early).astype(np.int32)


def lower_threshold_correctness_risk_v2(correctness, exit_points):
    exit_times = np.asarray(exit_points)
    correct_steps = np.asarray(correctness, dtype=bool)
    T = correct_steps.shape[0]
    suffix_sum = np.cumsum(correct_steps[::-1])[::-1].astype(np.float64)
    denom = np.arange(T, 0, -1, dtype=np.float64)  # T, T-1, ..., 1
    suffix_mean = suffix_sum / denom  # shape (T,)

    risks = np.zeros(exit_times.shape, dtype=np.float64)
    no_exit_val = len(correctness) - 1
    mask = (exit_times != no_exit_val)
    if np.any(mask):
        t = exit_times[mask].astype(int)
        if np.any(t < 0) or np.any(t >= T):
            bad = t[(t < 0) | (t >= T)]
            raise ValueError(f"Some exit_times out of range [0, {T-1}]: {bad[:10]}")
        risks[mask] = suffix_mean[t]

    return risks

def efficiency_loss_v1(correctness, exit_points):
    correctness = np.asarray(correctness, dtype=bool)
    T = correctness.shape[0]

    exit_points = np.asarray(exit_points)
    exit_points = exit_points.astype(int)

    # Edge case: empty trajectory
    if T == 0:
        return np.zeros_like(exit_points, dtype=float)

    no_exit = T - 1

    # Keep exits in-bounds
    e = np.clip(exit_points, 0, T - 1)

    # Sum of incorrect steps up to (and including) each time t
    incorrect = (~correctness).astype(np.int64)
    prefix_incorrect = np.cumsum(incorrect)  # shape (T,)
    numer = prefix_incorrect[e].astype(float)
    return numer / T


def efficiency_loss_v2(correctness, exit_points):
    correctness = np.asarray(correctness, dtype=bool)
    T = correctness.shape[0]

    exit_points = np.asarray(exit_points)
    exit_points = exit_points.astype(int)

    # Edge case: empty trajectory
    if T == 0:
        return np.zeros_like(exit_points, dtype=float)

    no_exit = T - 1
    # Keep exits in-bounds
    e = np.clip(exit_points, 0, T - 1)
    return e / T
    

def threshold_to_loss_vectorized_lower(
        entry,
        parameters,
        exit_time_fn,
        signal_field='confidence',
        correctness_field='intermediate_correct',
        signal_transform_fn=ema_causal,
        correctness_loss_fn=lower_threshold_correctness_risk_v2,
    ):
    assert len(thresholds.shape) == 1
    N = len(thresholds)
    signal_values = entry[signal_field]
    correctness = entry[correctness_field]
    if signal_transform_fn is not None:
        signal_values = signal_transform_fn(signal_values)
    lower_thresholds = sigmoid_func(
        entry['tokens_used'],
        **parameters
    )
    exit_time = exit_time_fn(signal_values, thresholds)  # (N,)
    exit_time = np.clip(exit_time, 0, correctness.size - 1)
    correctness_loss = lower_threshold_correctness_risk_v2(
        correctness, exit_time
    ).astype(np.float32)
    efficiency_loss = efficiency_loss_v1(
        correctness, exit_time
    ).astype(np.float32)
    return correctness_loss, efficiency_loss


def find_threshold(
        thresholds,
        correction_losses,
        efficiency_losses,
        epsilon,
        method='naive',
        B=1.0,
        delta=0.1
    ):
    n = correction_losses.shape[0]
    if method == 'naive':
        finite_sample_correction = 0.0
        adjusted_risk = correction_losses
    elif method == 'CRC':
        adjusted_risk = (n * correction_losses + B) / (n + 1)
    elif method == "UCB":
        concentration = np.sqrt(np.log(1 / max(delta, 1e-12)) / (2 * n))
        adjusted_risk = correction_losses + concentration
    elif method == 'LTT':
        raise NotImplementedError("LTT method not implemented yet.")
    else:
        raise ValueError(f"Unknown method: {method}")
    # Find all thresholds that satisfy the adjusted risk constraint
    valid_indices = np.where(adjusted_risk <= epsilon)[0]
    # Find the one with the best efficiency
    if len(valid_indices) == 0:
        return -1, (0, 1.0)
    best_idx = valid_indices[np.argmin(efficiency_losses[valid_indices])]
    return thresholds[best_idx], (correction_losses[best_idx], efficiency_losses[best_idx])
