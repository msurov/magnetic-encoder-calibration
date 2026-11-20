
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsqr
from scipy.linalg import null_space
from scipy.sparse import csr_array


def fit_periodic_bspline(x, y, k, nintervals, period) -> BSpline:
  x_normalized = np.mod(x, period)
  y = np.asarray(y, dtype=float)

  if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
    raise ValueError("x and y must be 1D arrays of equal length")

  if not (1 <= k <= 5):
    raise ValueError("k must be between 1 and 5")

  if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
    raise ValueError("x and y must contain finite values")

  h = period / nintervals
  t = h * np.arange(-k, nintervals + k + 1)

  A = BSpline.design_matrix(x_normalized, t, k)
  _, ncoefs = A.shape
  B = y
  C = np.zeros((k, ncoefs))
  for i in range(k):
    C[i,i] = 1
    C[i,i + nintervals] = -1

  N = null_space(C)

  A1 = A @ csr_array(N)
  c1,*_ = lsqr(A1, B)
  c = N @ c1

  return BSpline(t, c, k, extrapolate='periodic')
