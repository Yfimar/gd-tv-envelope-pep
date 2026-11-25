"""Design a closed-form certificate for ||g_N||^2 <= 1/(S+1)^2.

Identity (L=1,R=1), in basis [x_0, g_0..g_N] (m = N+2):
    nu*E00 - e_N e_N^T = sum_ij lam_ij M_ij + Q,  Q >> 0 (symmetric)
    + f-circulation: net f_t coefficient = 0 for all t (incl. star).
Tightness at Huber extremal (all g_t equal) forces support:
    lam_ij for iterate pairs i!=j in 0..N  (all tight),
    beta_t = lam_{*,t}  (tight),
    gamma = lam_{N,*} (tight);  lam_{t,*}=0 for t<N.
nu = 1/(S+1)^2 fixed.

M_ij defined so that I_ij(G) = <M_ij, G> = tr(M_ij G), M symmetric.
"""
import numpy as np
import cvxpy as cp


def interp_matrices(h):
    """Return dict (i,j)->M for all ordered pairs i!=j over nodes 0..N,-1(star).

    I_ij = f_i - f_j - <g_j, x_i - x_j> - 1/2||g_i - g_j||^2.
    Returns only the Gram (matrix) part as (m,m) symmetric arrays.
    """
    N = len(h)
    m = N + 2
    # coefficient vectors of x_t in basis: x_t = x_0 - sum_{s<t} h_s g_s
    cx = [None] * (N + 1)
    v = np.zeros(m); v[0] = 1.0
    for t in range(N + 1):
        cx[t] = v.copy()
        e = np.zeros(m); e[t + 1] = 1.0
        if t < N:
            v = v - h[t] * e
    eg = [None] * (N + 1)
    for t in range(N + 1):
        e = np.zeros(m); e[t + 1] = 1.0
        eg[t] = e
    zero = np.zeros(m)

    def M_of(i, j):
        # -<g_j, x_i - x_j> - 1/2||g_i - g_j||^2 as sym matrix
        cxi = cx[i] if i != -1 else zero
        cxj = cx[j] if j != -1 else zero
        cgi = eg[i] if i != -1 else zero
        cgj = eg[j] if j != -1 else zero
        dx = cxi - cxj
        dg = cgi - cgj
        M = -0.5 * (np.outer(cgj, dx) + np.outer(dx, cgj)) - 0.5 * np.outer(dg, dg)
        return M

    return M_of


def design(h, support="all", verbose=True):
    """Feasibility: find lam>=0 on allowed tight pairs, Q>>0,
    nu*E00 - F_N = sum lam M + Q, plus circulation constraints."""
    N = len(h)
    m = N + 2
    S = sum(h)
    nu = 1.0 / (S + 1) ** 2
    M_of = interp_matrices(h)

    pairs = []
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                pairs.append((i, j))
    for t in range(N + 1):
        pairs.append((-1, t))           # beta_t = lam_{*,t}
    pairs.append((N, -1))               # gamma = lam_{N,*}

    if support == "consec":
        keep = set()
        for t in range(N):
            keep.add((t, t + 1)); keep.add((t + 1, t))
        for t in range(N + 1):
            keep.add((-1, t))
        keep.add((N, -1))
        pairs = [p for p in pairs if p in keep]
    elif support == "toN":
        keep = set()
        for t in range(N):
            keep.add((t, N)); keep.add((N, t))
        for t in range(N + 1):
            keep.add((-1, t))
        keep.add((N, -1))
        pairs = [p for p in pairs if p in keep]
    elif support == "consec+toN":
        keep = set()
        for t in range(N):
            keep.add((t, t + 1)); keep.add((t + 1, t))
            if t < N:
                keep.add((t, N)); keep.add((N, t))
        for t in range(N + 1):
            keep.add((-1, t))
        keep.add((N, -1))
        pairs = [p for p in pairs if p in keep]

    lam = cp.Variable(len(pairs), nonneg=True)
    Q = cp.Variable((m, m), symmetric=True)

    Msum = sum(lam[k] * M_of(i, j) for k, (i, j) in enumerate(pairs))
    Enu = np.zeros((m, m)); Enu[0, 0] = nu
    FN = np.zeros((m, m)); FN[N + 1, N + 1] = 1.0
    cons = [Q >> 0, Enu - FN == Msum + Q]

    # f-circulation constraints
    # coeff of f_t in sum lam I_ij = out(t) - in(t)  [+lam_{t,*}, -beta_t]
    for t in range(N + 1):
        outt = sum(lam[k] for k, (i, j) in enumerate(pairs)
                   if i == t and j != -1)
        int_ = sum(lam[k] for k, (i, j) in enumerate(pairs)
                   if j == t and i != -1)
        beta = sum(lam[k] for k, (i, j) in enumerate(pairs)
                   if i == -1 and j == t)
        gam = sum(lam[k] for k, (i, j) in enumerate(pairs)
                  if i == t and j == -1)
        cons.append(outt + gam - int_ - beta == 0)
    # star node
    outstar = sum(lam[k] for k, (i, j) in enumerate(pairs) if i == -1)
    instar = sum(lam[k] for k, (i, j) in enumerate(pairs) if j == -1)
    cons.append(outstar - instar == 0)

    objective = cp.Minimize(cp.sum(lam)) if support.endswith("+l1") else cp.Minimize(0)
    if support.endswith("+l1"):
        support = support[:-3]
        # rebuild pairs under the underlying support
        allpairs = pairs
        pairs = []
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    pairs.append((i, j))
        for t in range(N + 1):
            pairs.append((-1, t))
        pairs.append((N, -1))
        if support == "consec":
            keep = set()
            for t in range(N):
                keep.add((t, t + 1)); keep.add((t + 1, t))
            for t in range(N + 1):
                keep.add((-1, t))
            keep.add((N, -1))
            pairs = [p for p in pairs if p in keep]
        elif support == "toN":
            keep = set()
            for t in range(N):
                keep.add((t, N)); keep.add((N, t))
            for t in range(N + 1):
                keep.add((-1, t))
            keep.add((N, -1))
            pairs = [p for p in pairs if p in keep]
        elif support == "consec+toN":
            keep = set()
            for t in range(N):
                keep.add((t, t + 1)); keep.add((t + 1, t))
                if t < N:
                    keep.add((t, N)); keep.add((N, t))
            for t in range(N + 1):
                keep.add((-1, t))
            keep.add((N, -1))
            pairs = [p for p in pairs if p in keep]
        lam = cp.Variable(len(pairs), nonneg=True)
        Msum = sum(lam[k] * M_of(i, j) for k, (i, j) in enumerate(pairs))
        cons = [Q >> 0, Enu - FN == Msum + Q]
        for t in range(N + 1):
            outt = sum(lam[k] for k, (i, j) in enumerate(pairs)
                       if i == t and j != -1)
            int_ = sum(lam[k] for k, (i, j) in enumerate(pairs)
                       if j == t and i != -1)
            beta = sum(lam[k] for k, (i, j) in enumerate(pairs)
                       if i == -1 and j == t)
            gam = sum(lam[k] for k, (i, j) in enumerate(pairs)
                      if i == t and j == -1)
            cons.append(outt + gam - int_ - beta == 0)
        outstar = sum(lam[k] for k, (i, j) in enumerate(pairs) if i == -1)
        instar = sum(lam[k] for k, (i, j) in enumerate(pairs) if j == -1)
        cons.append(outstar - instar == 0)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL)
    if verbose:
        print(f"h={h} support={support} status={prob.status}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    lamv = lam.value
    Qv = Q.value
    ev = np.linalg.eigvalsh(Qv)
    act = [(p, lamv[k]) for k, p in enumerate(pairs) if lamv[k] > 1e-7]
    if verbose:
        print(f"  mineig(Q)={ev[0]:.3e}  active={len(act)}/{len(pairs)}")
        for p, v in act:
            print(f"    lam{p} = {v:.6f}   *{(S+1)**2:.0f} = {v*(S+1)**2:.6f}")
    return dict(nu=nu, lam=lamv, Q=Qv, pairs=pairs, mineig=ev[0])


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    for sup in ["consec", "toN", "consec+toN", "all"]:
        print(f"===== support={sup} =====")
        design([1.0, 1.0], sup)
        print()
