"""Consolidated verification artifact for the terminal-gradient area law.

THEOREM. For L-smooth convex f, gradient descent x_{t+1}=x_t-(h_t/L) grad f(x_t)
with h_t in (0,1], ||x_0-x*|| <= R, S = sum_t h_t > 0:
        || grad f(x_N) || <= L R / (S + 1),   and this is tight.

Certificate identity (L=1, x*=0, f*=0):
  ||x_0||^2 - (S+1)^2 ||g_N||^2 =
      sum_t [U_t D_t + V_t E_t] + sum_t a_t F_t + b G
      + ||x_N - g_N||^2 + sum_s W_s ||g_s - g_{s+1}||^2
  D_t = f_t - f_{t+1} - h_t<g_{t+1},g_t> - (1/2)||g_t-g_{t+1}||^2 >= 0
  E_t = f_{t+1} - f_t + h_t||g_t||^2      - (1/2)||g_t-g_{t+1}||^2 >= 0
  F_t = -f_t + <g_t,x_t> - (1/2)||g_t||^2                        >= 0
  G   = f_N - (1/2)||g_N||^2                                    >= 0
  U_t = 2 S_{t+1}(S_{t+1}+1)/h_t,  V_t = 2 S_{t+1}(S_t+1)/h_t
  a_t = 2 h_t (t<N), a_N = 2,  b = 2(S+1)
  W_s = (S_{s+1}/h_s)(2(S_s+1) - h_s S_{s+1}) = V_s - S_{s+1}^2

Checks performed:
  [V1] symbolic: certificate identity for symbolic h, N = 1..4
  [V2] numeric (exact): identity residual == 0 for 200 random schedules
  [V3] numeric: SDP worst-case == 1/(S+1)^2 for random schedules h<=1,
       including schedules with zero steps
  [V4] numeric: tightness construction attains the bound
  [V5] numeric: W_s > 0 and residual Gram PSD for random h<=1 schedules
  [V6] numeric: boundary -- overshoot lower bound, single step h>sqrt2 fails
  [V7] numeric: zero-step schedules reduce to compressed schedule
"""
import numpy as np
import sympy as sp


# ---------- closed-form weights ----------
def weights(h):
    h = np.asarray(h, float)
    N = len(h)
    S = h.sum()
    Sp = np.concatenate([[0.0], np.cumsum(h)])          # S_t, t=0..N
    U = 2 * Sp[1:] * (Sp[1:] + 1) / h
    V = 2 * Sp[1:] * (Sp[:-1] + 1) / h
    a = np.concatenate([2 * h, [2.0]])
    b = 2 * (S + 1)
    W = (Sp[1:] / h) * (2 * (Sp[:-1] + 1) - h * Sp[1:])
    return U, V, a, b, W, Sp, S


# ---------- interpolation matrices over basis [x_0, g_0..g_N] ----------
from cert_design import interp_matrices as _raw_M


def _Mparts(h):
    """M(i,j) s.t. <M(i,j),G> = -<g_j,x_i-x_j>-.5||g_i-g_j||^2 + fpart."""
    N = len(h)
    M_of = _raw_M(h)

    def M(i, j):
        fpart = np.zeros(N + 1)
        if i >= 0: fpart[i] += 1.0
        if j >= 0: fpart[j] -= 1.0
        return M_of(i, j), fpart
    return M


def residual_gram(h):
    """Residual Q = E_X - (S+1)^2 F_N - sum lambda M."""
    N = len(h); m = N + 2
    U, V, a, b, W, Sp, S = weights(h)
    M = _Mparts(h)
    E = np.zeros((m, m)); E[0, 0] = 1.0
    Fn = np.zeros((m, m)); Fn[N + 1, N + 1] = (S + 1) ** 2
    Q = E - Fn
    fres = np.zeros(N + 1)
    for t in range(N):
        Mt, ft = M(t, t + 1); Q -= U[t] * Mt; fres += U[t] * ft
        Mt, ft = M(t + 1, t); Q -= V[t] * Mt; fres += V[t] * ft
    for t in range(N + 1):
        Mt, ft = M(-1, t); Q -= a[t] * Mt; fres += a[t] * ft
    Mt, ft = M(N, -1); Q -= b * Mt; fres += b * ft
    return Q, fres


def closed_gram(h):
    """Closed-form Q = ||x_N - g_N||^2 + sum_s W_s ||g_s-g_{s+1}||^2."""
    N = len(h); m = N + 2
    U, V, a, b, W, Sp, S = weights(h)
    v = np.zeros(m); v[0] = 1.0
    for s in range(N): v[s + 1] = -h[s]
    v[N + 1] -= 1.0                                # x_N - g_N
    Q = np.outer(v, v)
    for s in range(N):
        d = np.zeros(m); d[s + 1] = 1.0; d[s + 2] = -1.0
        Q += W[s] * np.outer(d, d)
    return Q


# ---------- SDP ----------
def sdp_worstcase(h):
    """max ||g_N||^2 s.t. interpolation + ||x_0||<=1 (dim-free SDP)."""
    from pep_sdp import solve_pep
    return solve_pep(list(h), "gN")["value"]


# ---------- symbolic identity ----------
def symbolic_check(N):
    hs = sp.symbols('h0:%d' % N, positive=True)
    fs = sp.symbols('f0:%d' % (N + 1))
    ip = {}
    for i in range(N + 2):
        for j in range(i, N + 2):
            ip[i, j] = sp.Symbol('I%d_%d' % (i, j))

    def inner(u, v):
        return sum(a_ * b_ * ip[min(i, j), max(i, j)]
                   for i, a_ in u.items() for j, b_ in v.items())

    x = [None] * (N + 1); g = [None] * (N + 1)
    x[0] = {0: 1}
    for t in range(N + 1):
        g[t] = {t + 1: 1}
    for t in range(1, N + 1):
        x[t] = dict(x[t - 1])
        for s, c in g[t - 1].items():
            x[t][s] = x[t].get(s, 0) - hs[t - 1] * c
    S = sum(hs)
    Sp = [0] * (N + 1)
    for t in range(N):
        Sp[t + 1] = Sp[t] + hs[t]
    U = [2 * Sp[t + 1] * (Sp[t + 1] + 1) / hs[t] for t in range(N)]
    V = [2 * Sp[t + 1] * (Sp[t] + 1) / hs[t] for t in range(N)]
    a = [2 * hs[t] for t in range(N)] + [2]
    b = 2 * (S + 1)
    W = [(Sp[s + 1] / hs[s]) * (2 * (Sp[s] + 1) - hs[s] * Sp[s + 1])
         for s in range(N)]

    def sub(u, v):
        return {k: u.get(k, 0) - v.get(k, 0) for k in set(u) | set(v)}

    rhs = 0
    for t in range(N):
        d = sub(g[t], g[t + 1])
        Dt = fs[t] - fs[t + 1] - hs[t] * inner(g[t + 1], g[t]) \
            - sp.Rational(1, 2) * inner(d, d)
        Et = fs[t + 1] - fs[t] + hs[t] * inner(g[t], g[t]) \
            - sp.Rational(1, 2) * inner(d, d)
        rhs += U[t] * Dt + V[t] * Et
    for t in range(N + 1):
        Ft = -fs[t] + inner(g[t], x[t]) - sp.Rational(1, 2) * inner(g[t], g[t])
        rhs += a[t] * Ft
    rhs += b * (fs[N] - sp.Rational(1, 2) * inner(g[N], g[N]))
    z = sub(x[N], g[N]); rhs += inner(z, z)
    for s in range(N):
        d = sub(g[s], g[s + 1]); rhs += W[s] * inner(d, d)
    lhs = inner(x[0], x[0]) - (S + 1) ** 2 * inner(g[N], g[N])
    return sp.expand(rhs - lhs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = True

    print("== [V1] symbolic identity (symbolic h) ==")
    for N in range(1, 5):
        r = symbolic_check(N)
        print(f"  N={N}: residual = {r}")
        ok &= (r == 0)

    print("== [V2] numeric identity residual ==")
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(1, 8))
        h = rng.uniform(0.05, 1.0, n)
        Q, fres = residual_gram(h)
        worst = max(worst, np.abs(Q - closed_gram(h)).max(),
                    np.abs(fres).max())
    print(f"  max |residual| over 200 schedules = {worst:.3e}")
    ok &= worst < 1e-9

    print("== [V3] SDP worst case == formula ==")
    scheds = [[1.0], [0.5], [1, 1], [0.5, 0.5], [1, 1, 1],
              [1, 0.5, 0.7], [0.9, 0.7, 0.5, 0.3], [0.2] * 5,
              [1, 0, 1], [0.5, 0, 0.5, 0], [0, 0.7, 0.3, 0]]
    for _ in range(12):
        scheds.append(list(rng.uniform(0.0, 1.0, int(rng.integers(1, 7)))))
    for h in scheds:
        h = np.asarray(h, float)
        S = h.sum()
        if S < 1e-9:
            continue
        v = sdp_worstcase(h)
        pred = 1.0 / (S + 1) ** 2
        flag = "OK" if abs(v - pred) < 3e-6 else "MISMATCH"
        print(f"  h={np.round(h,3)} S={S:.3f} SDP={v:.6f} pred={pred:.6f} {flag}")
        ok &= abs(v - pred) < 3e-6

    print("== [V4] Huber tightness ==")
    for h in [[1.0], [0.5, 0.8, 1.0], [1, 1, 1], [0.3] * 4, [1, 0, 1]]:
        h = np.asarray(h, float); S = h.sum()
        a = 1.0 / (S + 1)                        # R = L = 1
        xt, gN = 1.0, a                          # x_0 = R = 1, g_t = a
        for t in range(len(h)):
            xt = xt - h[t] * a
        print(f"  h={np.round(h,3)}: x_N={xt:.6f} a={a:.6f} |g_N|={a:.6f}"
              f" vs 1/(S+1)={1/(S+1):.6f}")
        ok &= abs(xt - a) < 1e-12 and abs(a - 1 / (S + 1)) < 1e-12

    print("== [V5] positivity / PSD residual ==")
    minw = 1e9; mineig = 1e9
    for _ in range(100):
        n = int(rng.integers(1, 8))
        h = rng.uniform(0.02, 1.0, n)
        U, V, a, b, W, Sp, S = weights(h)
        minw = min(minw, W.min())
        mineig = min(mineig, np.linalg.eigvalsh(closed_gram(h))[0])
    print(f"  min W_s = {minw:.4f} (>0),  min eigenvalue residual = {mineig:.3e}")
    ok &= minw > 0 and mineig > -1e-9

    print("== [V6] boundary h>1 ==")
    # single step: worst case = max{1/(1+h)^2, (h-1)^2}, cross at sqrt2
    for h in [1.4, 1.42, 1.5]:
        v = sdp_worstcase([h])
        pred = max(1 / (1 + h) ** 2, (h - 1) ** 2)
        print(f"  h={h}: SDP={v:.6f} max-branches={pred:.6f}")
        ok &= abs(v - pred) < 3e-6
    # overshoot lower bound (h0-1)/(sig1+1) for [1.8,0.3]
    v = sdp_worstcase([1.8, 0.3])
    lb = (1.8 - 1) / (0.3 + 1)
    print(f"  h=[1.8,0.3]: SDP={v:.6f} overshoot LB^2={lb**2:.6f}")
    ok &= abs(v - lb ** 2) < 3e-6
    # certificate edge sign
    U, V, a, b, W, Sp, S = weights([1.8, 0.3])
    print(f"  h=[1.8,0.3]: W = {np.round(W,4)} (negative => cert fails)")
    ok &= W[0] < 0

    print("== [V7] zero-step reduction ==")
    for h in [[1, 0, 1], [0.5, 0, 0.5], [0, 1, 0.5, 0]]:
        h = np.asarray(h, float)
        S = h.sum()
        v = sdp_worstcase(h)
        pred = 1.0 / (S + 1) ** 2
        print(f"  h={h}: S={S} SDP={v:.6f} pred={pred:.6f}")
        ok &= abs(v - pred) < 3e-6

    print("== [V8] envelope h_s <= h_TV(S_s) and uniform [0,sqrt2]^N ==")

    def h_tv(ss):
        return (-ss + np.sqrt(ss ** 2 + 8 * (ss + 1))) / 2

    # 8a: random schedules in [0,sqrt2]^N -> SDP == 1/(S+1)^2, W >= 0
    for _ in range(6):
        n = int(rng.integers(1, 6))
        h = rng.uniform(0.0, np.sqrt(2), n)
        S = h.sum()
        if S < 1e-9:
            continue
        v = sdp_worstcase(h)
        pred = 1.0 / (S + 1) ** 2
        U, V, a, b, W, Sp, _ = weights(h)
        flag = "OK" if abs(v - pred) < 3e-6 and W.min() >= -1e-9 else "MISMATCH"
        print(f"  sqrt2-box h={np.round(h,3)} SDP={v:.6f} pred={pred:.6f}"
              f" minW={W.min():.4f} {flag}")
        ok &= abs(v - pred) < 3e-6 and W.min() >= -1e-9

    # 8b: envelope schedules with late steps approaching 2
    env_scheds = [[1.0] * 8 + [1.8],        # 1.8 <= h_TV(8) = 1.831..
                  [1.0] * 4 + [1.6],        # 1.6 <= h_TV(4) = 1.690..
                  [1.2, 1.3, 1.4, 1.5],     # check each step below env
                  [np.sqrt(2)],             # exact boundary N=1
                  [np.sqrt(2), np.sqrt(2)]]
    for h in env_scheds:
        h = np.asarray(h, float)
        ss = 0.0; inside = True
        for s in range(len(h)):
            inside &= h[s] <= h_tv(ss) + 1e-12
            ss += h[s]
        S = h.sum()
        v = sdp_worstcase(h)
        pred = 1.0 / (S + 1) ** 2
        U, V, a, b, W, Sp, _ = weights(h)
        flag = "OK" if abs(v - pred) < 3e-6 and W.min() >= -1e-9 else "MISMATCH"
        print(f"  envelope h={np.round(h,3)} inside={inside} SDP={v:.6f}"
              f" pred={pred:.6f} minW={W.min():.4f} {flag}")
        ok &= inside and abs(v - pred) < 3e-6 and W.min() >= -1e-9

    # 8c: the exact TV equality curve: W_s == 0 identically
    h = []
    ss = 0.0
    for _ in range(4):
        hs = h_tv(ss)
        h.append(hs)
        ss += hs
    U, V, a, b, W, Sp, S = weights(np.asarray(h))
    print(f"  TV equality curve h={np.round(h,4)}: max|W|={np.abs(W).max():.3e}"
          f" (=0 on boundary)")
    ok &= np.abs(W).max() < 1e-9

    print("\nALL CHECKS", "PASS" if ok else "FAIL")
