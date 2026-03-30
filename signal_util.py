import numpy as np
import json



def load_log(file_dir):
    with open(file_dir, 'r') as f:
        data = json.load(f)

    question_entries = []
    iterable = data.values() if isinstance(data, dict) else data
    for item in iterable:
        if not isinstance(item, dict):
            continue
        
        checks = item.get('intermediate_checks') or item.get('checks')
        if not checks and 'checkpoints' in item:
            checkpoints = item.get('checkpoints', [])
            checks = []
            for cp in checkpoints:
                check = {
                    'thinking_step': cp.get('checkpoint_idx', cp.get('tokens_used', 0)),
                    'predicted_answer': cp.get('predicted_answer'),
                    'confidence': cp.get('confidence'),
                    'intermediate_correct': cp.get('is_correct'),
                    'tokens_used': cp.get('tokens_used', 0)
                }
                # Copy signal fields
                for field in ['entropy_uni', 'eat_uni', 'eat_uni_forced', 'probe_confidence', 'entropy', 'tokens_used']:
                    if field in cp:
                        check[field] = cp[field]
                checks.append(check)
        
        if checks:
            for chk in checks:
                if 'intermediate_correct' not in chk and 'is_correct' in chk:
                    chk['intermediate_correct'] = chk['is_correct']
            
            item['intermediate_checks'] = checks
            # Transfer each subfiled in checkpoints into a numpy array
            for field in checks[0].keys():
                item[field] = np.array([chk.get(field) for chk in checks])

            question_entries.append(item)
    return question_entries


def exit_time_first_geq(signal_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """First t with signal[t] >= threshold, else T-1. Vectorized over thresholds."""
    s = np.asarray(signal_values)
    th = np.asarray(thresholds)
    T = s.size
    if T == 0:
        return np.zeros_like(th, dtype=np.int64)

    prefix_max = np.maximum.accumulate(s)                 # nondecreasing
    idx = np.searchsorted(prefix_max, th, side="left")    # [0..T]
    return np.minimum(idx, T - 1).astype(np.int64)


def exit_time_first_leq(signal_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """First t with signal[t] <= threshold, else T-1. Vectorized over thresholds."""
    s = np.asarray(signal_values)
    th = np.asarray(thresholds)
    T = s.size
    if T == 0:
        return np.zeros_like(th, dtype=np.int64)

    prefix_min = np.minimum.accumulate(s)                 # nonincreasing
    # Make it nondecreasing via negation so we can use searchsorted
    idx = np.searchsorted(-prefix_min, -th, side="left")  # [0..T]
    return np.minimum(idx, T - 1).astype(np.int64)


def exit_time_lower_threshold(
    signal_values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    s = np.asarray(signal_values)
    th = np.asarray(thresholds)

    if s.ndim != 1:
        raise ValueError(f"signal_values must be 1D (N,), got shape {s.shape}")
    if th.ndim != 2:
        raise ValueError(f"thresholds must be 2D (M, N), got shape {th.shape}")
    if th.shape[1] != s.shape[0]:
        raise ValueError(
            f"thresholds.shape[1] must equal len(signal_values): "
            f"{th.shape[1]} != {s.shape[0]}"
        )

    N = s.shape[0]
    if N == 0:
        return np.zeros(th.shape[0], dtype=np.int64)

    crossed = (s[None, :] <= th)
    any_cross = crossed.any(axis=1)
    first_idx = crossed.argmax(axis=1)

    exit_pos = np.where(any_cross, first_idx, N - 1)
    return exit_pos.astype(np.int64)


def ema_causal(x, alpha=0.95) -> np.ndarray:
    """
    y[t] = alpha*x[t] + (1-alpha)*y[t-1], with y[0]=x[0]
    alpha in (0,1]. Larger alpha = less smoothing.
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.size
    if T == 0:
        return x

    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    one_minus = 1.0 - float(alpha)
    a = float(alpha)
    for t in range(1, T):
        y[t] = a * x[t] + one_minus * y[t - 1]
    return y
