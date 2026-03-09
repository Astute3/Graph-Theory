#!/usr/bin/env python3
"""
Minimum domination number γ(G(n,2)) for generalized Petersen graphs G(n,2),
for n = 5 to 25.

G(n,2) has 2n vertices:
  outer: u_0 ... u_{n-1}   (indices 0 .. n-1)
  inner: v_0 ... v_{n-1}   (indices n .. 2n-1, v_i stored at n+i)

Edges:
  outer cycle  : u_i — u_{(i+1) mod n}
  inner "star" : v_i — v_{(i+2) mod n}
  spokes       : u_i — v_i

Strategy
--------
  1. Validate against known γ values for n = 5..10 using brute-force BFS
     (with rotational-symmetry reduction: fix vertex 0 always in the set).
  2. Compute γ for all n = 5..25 using an exact ILP formulation solved with
     scipy.optimize.milp (Branch-and-Bound under the hood).
"""

import sys
import itertools
from math import ceil

# ─── Graph construction & bitmask utilities ──────────────────────────────────

def build_cover_masks(n):
    """
    Return (cover, full_mask) for G(n,2).

    cover[v]  = bitmask with bit v and every neighbor of v set.
    full_mask = (1 << 2n) - 1  (all vertices covered).
    """
    N = 2 * n
    cover = [0] * N

    for i in range(n):
        # outer vertex i: cycle neighbors (i±1 mod n) + spoke to n+i
        v = i
        m = 1 << v
        for j in [(i - 1) % n, (i + 1) % n, n + i]:
            m |= 1 << j
        cover[v] = m

        # inner vertex n+i: spoke to i + two star neighbors (i±2 mod n shifted by n)
        v = n + i
        m = 1 << v
        for j in [i, n + (i + 2) % n, n + (i - 2) % n]:
            m |= 1 << j
        cover[v] = m

    return cover, (1 << N) - 1


def is_dominating(subset, cover, full_mask):
    """True iff every vertex is covered by at least one member of subset."""
    result = 0
    for v in subset:
        result |= cover[v]
    return result == full_mask


# ─── Brute-force BFS solver (validation / small n) ───────────────────────────

def find_gamma_brute(n, k_min=None, k_max=None):
    """
    Enumerate subsets of increasing size k (BFS layers) until a dominating
    set is found.  Uses rotational symmetry of G(n,2): vertex 0 is fixed in
    every candidate set, reducing the search space by roughly a factor of 2n.

    Returns (gamma, example_set).
    """
    cover, full_mask = build_cover_masks(n)
    N = 2 * n
    # Lower bound: each vertex covers at most 4 (itself + 3 neighbours)
    if k_min is None:
        k_min = ceil(n / 2)
    # Safe upper bound (always achievable constructively)
    if k_max is None:
        k_max = n

    rest = list(range(1, N))           # vertices other than 0
    for k in range(k_min, k_max + 1):
        for tail in itertools.combinations(rest, k - 1):
            s = (0,) + tail
            if is_dominating(s, cover, full_mask):
                return k, list(s)
    return None, None


# ─── Exact ILP solver (scipy.optimize.milp) ──────────────────────────────────

def find_gamma_ilp(n):
    """
    Solve the minimum dominating set as an ILP:

      minimize   sum_v x_v
      subject to for every vertex v:
                   sum_{u : v in closed_N(u)} x_u  >= 1
                 x_v in {0, 1}

    Returns (gamma, example_set) or (None, None) on failure.
    """
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    cover, _ = build_cover_masks(n)
    N = 2 * n

    # A[v, u] = 1  iff  u "covers" v  (i.e. bit v is set in cover[u])
    A = np.zeros((N, N), dtype=np.float64)
    for u in range(N):
        for v in range(N):
            if (cover[u] >> v) & 1:
                A[v, u] = 1.0

    c           = np.ones(N, dtype=np.float64)
    constraints = LinearConstraint(A, lb=np.ones(N), ub=np.full(N, np.inf))
    integrality = np.ones(N, dtype=int)          # all binary
    var_bounds  = Bounds(lb=np.zeros(N), ub=np.ones(N))

    res = milp(c, constraints=constraints, integrality=integrality,
               bounds=var_bounds)

    if res.success:
        x       = np.round(res.x).astype(int)
        gamma   = int(x.sum())
        dom_set = [i for i in range(N) if x[i] == 1]
        return gamma, dom_set

    return None, None


# ─── Validation ──────────────────────────────────────────────────────────────

KNOWN = {5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6}


def validate():
    """
    Verify the graph construction and domination check against published
    values.  Aborts with exit code 1 if any case is wrong.
    """
    print("=== Validating against known γ values ===")
    ok = True
    for n, expected in sorted(KNOWN.items()):
        gamma, dom_set = find_gamma_brute(n)
        flag = "OK" if gamma == expected else f"FAIL (expected {expected})"
        print(f"  G({n:2d},2): gamma = {gamma}  {flag}  example set = {dom_set}")
        if gamma != expected:
            ok = False
    print()
    if not ok:
        print("VALIDATION FAILED — bug in graph construction or domination check.")
        sys.exit(1)
    print("All validation cases passed.\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    validate()

    print("=== gamma(G(n,2)) for n = 5 to 25 ===")
    print()
    print(f"{'n':>4}  {'gamma':>5}  min dominating set")
    print("-" * 70)

    for n in range(5, 51):
        gamma, dom_set = find_gamma_ilp(n)
        if gamma is None:
            print(f"{n:>4}  {'?':>5}  (ILP solver failed)")
        else:
            print(f"{n:>4}  {gamma:>5}  {dom_set}")


if __name__ == "__main__":
    main()
