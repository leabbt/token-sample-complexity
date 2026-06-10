"""Token sampling strategies used by the Monte-Carlo experiments."""

from __future__ import annotations

import torch


def sample_iid(X: torch.Tensor, n: int, replace: bool = True) -> torch.Tensor:
    """Draw `n` tokens i.i.d. from X (shape [batch, seq_len, d])."""
    if replace:
        idx = torch.randint(0, X.size(1), (n,), device=X.device)
    else:
        idx = torch.randperm(X.size(1), device=X.device)[:n]
    return X[:, idx, :]


def sample_window_and_random(
    X: torch.Tensor,
    w: int,
    r: int,
    query_pos: int | None = None,
) -> torch.Tensor:
    """Take a centred window of half-width `w` plus `r` random tokens from outside it.

    Effective sample size: `n = (2w + 1) + r_actual`, where `r_actual` is reduced
    when fewer than `r` tokens sit outside the window. The selected indices are
    returned in increasing order.
    """
    N = X.size(1)
    if query_pos is None:
        query_pos = N // 2

    win_start = max(0, query_pos - w)
    win_end = min(N, query_pos + w + 1)
    window_idx = torch.arange(win_start, win_end, device=X.device)

    if r > 0:
        all_idx = torch.arange(N, device=X.device)
        outside_mask = torch.ones(N, dtype=torch.bool, device=X.device)
        outside_mask[win_start:win_end] = False
        outside_idx = all_idx[outside_mask]
        r_actual = min(r, outside_idx.size(0))
        if r_actual > 0:
            perm = torch.randperm(outside_idx.size(0), device=X.device)[:r_actual]
            selected = torch.cat([window_idx, outside_idx[perm]])
        else:
            selected = window_idx
    else:
        selected = window_idx

    selected = selected.sort().values
    return X[:, selected, :]
