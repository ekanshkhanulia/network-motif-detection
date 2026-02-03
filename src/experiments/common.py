from __future__ import annotations

from typing import List


def build_p_schedule(k: int, q: float, schedule: str) -> List[float]:
    """Construct depth-wise probabilities p_d for RAND-ESU such that prod p_d ~= q.

    Schedules (per Wernicke 2005 Section 2.2 and Figure 3):
      - fine: (1, 1, ..., 1, q) - Article's "fine" variant, p=1 until last depth
      - coarse: (1, 1, ..., sqrt(q), sqrt(q)) - Article's "coarse" variant
      - geometric: p_d = q**(1/k) for all depths (uniform)
      - skewed: p1=1.0, p_{2..k} = q**(1/(k-1)) (our variant, reduces variance)

    The article recommends "fine" for better sampling quality (Figure 3b).
    """
    if not (0 < q <= 1):
        raise ValueError("q must be in (0,1]")
    if k <= 0:
        raise ValueError("k must be >= 1")

    if schedule == "fine":
        # Article's "fine" variant: (1, 1, ..., 1, q)
        # All depths have p=1 except the last which has p=q
        if k == 1:
            return [q]
        return [1.0] * (k - 1) + [q]
    elif schedule == "coarse":
        # Article's "coarse" variant: (1, 1, ..., sqrt(q), sqrt(q))
        # Last two depths have sqrt(q)
        if k == 1:
            return [q]
        if k == 2:
            return [q**0.5, q**0.5]
        return [1.0] * (k - 2) + [q**0.5, q**0.5]
    elif schedule == "geometric":
        p = q ** (1.0 / k)
        return [p for _ in range(k)]
    elif schedule == "skewed":
        if k == 1:
            return [q]
        p_rest = q ** (1.0 / (k - 1))
        return [1.0] + [p_rest for _ in range(k - 1)]
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")
