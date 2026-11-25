"""PEP SDP for gradient descent with a fixed step-size schedule on
L-smooth convex functions (F_{0,L}), normalized to L = 1, R = 1.

Iterate: x_{i+1} = x_i - h_i g_i,  i = 0..N-1   (h_i = L * stepsize, dim'less)
Data set: {(x_i, g_i, f_i)}_{i in {0..N} u {*}} with x_* = g_* = f_* = 0.
Interpolability (Taylor-Hendrickx-Glineur, necessary & sufficient):
    f_i - f_j >= <g_j, x_i - x_j> + (1/2) ||g_i - g_j||^2   (i != j)
Initial condition: ||x_0||^2 <= 1.
Objective choices: f_N  |  G(x_N,x_N)=||x_N||^2  |  G(g_N,g_N)=||g_N||^2 | mean f_i.
"""
import numpy as np
import cvxpy as cp


def build_coeffs(h):
    """Return coefficient maps for x_i and g_i in basis [x_0, g_0,...,g_N]."""
    N = len(h)
    m = N + 2
    cx = []          # cx[i] : coefficients of x_i
    cg = []          # cg[i] : coefficients of g_i
    v = np.zeros(m); v[0] = 1.0           # x_0
    for i in range(N + 1):
        cx.append(v.copy())
        e = np.zeros(m); e[i + 1] = 1.0   # g_i
        cg.append(e)
        if i < N:
            v = v - h[i] * e              # x_{i+1} = x_i - h_i g_i
    return cx, cg, m


def solve_pep(h, measure="fN", solver=cp.CLARABEL, tol=1e-9):
    """Exact worst-case of the chosen measure over F_{0,1}, ||x_0|| <= 1.

    Returns dict(value, G, f, prob, cons) where cons lists the
    interpolation constraints (for dual extraction).
    """
    N = len(h)
    cx, cg, m = build_coeffs(h)
    G = cp.Variable((m, m), symmetric=True)
    f = cp.Variable(N + 1)

    def IP(ca, cb):
        return ca @ G @ cb

    cons = [G >> 0, IP(cx[0], cx[0]) <= 1.0]
    interp_cons = []
    idx = list(range(N + 1)) + [-1]        # -1 encodes *
    zero = np.zeros(m)
    for i in idx:
        for j in idx:
            if i == j:
                continue
            fi = f[i] if i != -1 else 0.0
            fj = f[j] if j != -1 else 0.0
            cgi = cg[i] if i != -1 else zero
            cgj = cg[j] if j != -1 else zero
            cxi = cx[i] if i != -1 else zero
            cxj = cx[j] if j != -1 else zero
            lhs = fi - fj - IP(cgj, cxi - cxj) - 0.5 * IP(cgi - cgj, cgi - cgj)
            c = lhs >= 0
            interp_cons.append((i, j, c))
            cons.append(c)

    if measure == "fN":
        obj = f[N]
    elif measure == "distN":
        obj = IP(cx[N], cx[N])
    elif measure == "gN":
        obj = IP(cg[N], cg[N])
    elif measure == "fmean":
        obj = cp.sum(f) / (N + 1)
    else:
        raise ValueError(measure)
    prob = cp.Problem(cp.Maximize(obj), cons)
    val = prob.solve(solver=solver)
    return dict(value=float(val), G=G.value, f=f.value, prob=prob,
                interp=interp_cons, radius_cons=cons[1], cx=cx, cg=cg)


def solve_pep_fast(h, measure="fN", solver=cp.CLARABEL):
    r = solve_pep(h, measure, solver)
    return r["value"]


if __name__ == "__main__":
    # sanity: constant step h=1, small N
    for N in [1, 2, 3, 5, 10]:
        h = [1.0] * N
        print(f"N={N:2d}  const h=1:  fN = {solve_pep_fast(h):.8f}")
