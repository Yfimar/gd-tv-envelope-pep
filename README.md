# Verification code: exact terminal gradient bounds for gradient descent within the Teboulle--Vaisbourd envelope

Accompanies the paper *Exact terminal gradient bounds for gradient descent within the Teboulle--Vaisbourd envelope* (M. Li).

## Install

```
pip install -r requirements.txt
```

## Run

```
python verify_artifact.py
```

## Expected output

All checks pass, ending with `ALL CHECKS PASS`:

- symbolic verification of the certificate identity in the step sizes, N = 1..4;
- numerical identity residuals ~1e-13 over 200 random schedules;
- SDP worst-case values matching 1/(S+1)^2 on random schedules in [0,1]^N, in [0,sqrt(2)]^N, and inside the dynamic envelope h_s(S_s+h_s) <= 2(S_s+1), including steps approaching 2 and zero steps;
- exactness of the Huber tightness construction;
- nonnegativity of the edge weights inside the envelope, vanishing exactly on the Teboulle--Vaisbourd equality curve;
- boundary behavior: single-step crossover at sqrt(2) and the overshoot lower bound.
