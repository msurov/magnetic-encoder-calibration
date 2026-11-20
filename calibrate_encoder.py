import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import median_filter
from smoothing_periodic_bspline import fit_periodic_bspline
from dataclasses import dataclass
from typing import Literal, Tuple
import argparse
import json


@dataclass
class CalibData:
  time : np.ndarray
  angle : np.ndarray

@dataclass
class CalibDataExt(CalibData):
  normalized_angle : np.ndarray
  direction : Literal['positive', 'negative']
  points_per_revolute : int

@dataclass
class EncoderParameters:
  spline : BSpline
  points_per_revolute : int

@dataclass
class LookupTableFitResult(EncoderParameters):
  normalized_angle : np.ndarray
  deviation : np.ndarray
  fitting_error : float
  mask : np.ndarray

def unwrap_angle(arr : np.ndarray, period : float) -> np.ndarray:
  n = len(arr)
  result = np.copy(arr)
  nturns = 0

  for i in range(1, n):
    nturns -= np.round((arr[i] - arr[i-1]) / period).astype(int)
    result[i] = arr[i] + nturns * period

  return result

def load_data(logpath : str) -> CalibData:
  data = np.genfromtxt(logpath, delimiter=',', names=True)
  return CalibData(
    time = data['time'],
    angle = data['angle']
  )

def fit_periodic_spline_stable(x : np.ndarray, y : np.ndarray, period : float, k : int, nintervals : int) -> Tuple[BSpline, np.ndarray]:
  sp = fit_periodic_bspline(x, y, period=period, k=k, nintervals=nintervals)
  d = np.abs(sp(x) - y)
  eps = 3 * np.median(d)
  mask = d < eps
  sp = fit_periodic_bspline(x[mask], y[mask], period=period, k=k, nintervals=nintervals)
  return sp, mask

def prepare_data(data : CalibData, encoder_resolution : int) -> CalibDataExt:
  maxval = 2**encoder_resolution
  angle = data.angle / maxval
  unwrapped = unwrap_angle(angle, 1.)
  direction = 'positive' if unwrapped[-1] > unwrapped[0] else 'negative'
  return CalibDataExt(
    time = data.time, 
    angle = data.angle, 
    normalized_angle = unwrapped,
    direction = direction,
    points_per_revolute = maxval
  )

def load_encoder_parameters(path : str) -> EncoderParameters:
  with open(path, 'r') as f:
    data = json.load(f)
    spline_entry = data['spline']
    sp = BSpline(
      t = np.asarray(spline_entry['knots']),
      c = np.asarray(spline_entry['controls']),
      k = int(spline_entry['degree'])
    )
    if spline_entry['type'] == 'periodic-bspline':
      sp.extrapolate = 'periodic'
    else:
      sp.extrapolate = None
    return EncoderParameters(
      spline = sp,
      points_per_revolute = int(data['points_per_revolute'])
    )

def save_encoder_parameters(path : str, encpar : EncoderParameters):
  config = {
    'spline': {
      'knots': encpar.spline.t.tolist(),
      'controls': encpar.spline.c.tolist(),
      'degree': encpar.spline.k,
      'period': 1.0,
      'type': 'periodic-bspline'
    },
    'points_per_revolute': encpar.points_per_revolute
  }
  with open(path, 'w') as f:
    json.dump(config, f, indent=2)

def fit_lookup_table(data : CalibDataExt, nintervals : int) -> LookupTableFitResult:
  match data.direction:
    case 'positive': invsp = make_interp_spline(data.normalized_angle, data.time, k=1)
    case 'negative': invsp = make_interp_spline(data.normalized_angle[::-1], data.time[::-1], k=1)

  invsp.extrapolate = None
  period = invsp(data.normalized_angle + 0.5) - invsp(data.normalized_angle - 0.5)
  period = median_filter(period, 9)

  # error estimation
  sp = make_interp_spline(data.time, data.normalized_angle, k=1)
  Isp = sp.antiderivative()
  deviation = data.normalized_angle - (Isp(data.time + period/2) - Isp(data.time - period/2)) / period

  # filter out valid data
  mask1 = np.isfinite(deviation)
  x = data.normalized_angle[mask1] % 1.
  y = deviation[mask1]

  # fit data by periodic BSpline
  table, mask2 = fit_periodic_spline_stable(x, y, period=1., k=3, nintervals=nintervals)
  mask1[mask1] = mask2
  fitting_error = np.mean(np.abs(table(x[mask2]) - y[mask2]))

  return LookupTableFitResult(
    spline = table,
    normalized_angle = data.normalized_angle,
    deviation = deviation,
    mask = mask1,
    fitting_error = fitting_error,
    points_per_revolute = data.points_per_revolute
  )

def plot_fitted(fit_res : LookupTableFitResult):
  mask = fit_res.mask
  a = fit_res.normalized_angle
  e = fit_res.deviation
  x = np.linspace(0., 1., 300)
  y = fit_res.spline(x)
  plt.grid(True)
  plt.plot(a[~mask] % 1., e[~mask], '.', alpha=0.2, color='red')
  plt.plot(a[mask] % 1., e[mask], '.', alpha=0.2, color='green')
  plt.plot(x, y, color='orange')
  plt.tight_layout(pad=0.1)

def undistort_angle(x : np.ndarray, encpar : EncoderParameters) -> np.ndarray:
  y = x - encpar.spline(x / encpar.points_per_revolute) * encpar.points_per_revolute
  return y

def plot_velocity(data : CalibDataExt, fit_res : LookupTableFitResult):
  x = data.normalized_angle - fit_res.spline(data.normalized_angle)
  t = data.time
  v = np.gradient(x, t)
  plt.grid(True)
  plt.plot(t, v)
  plt.tight_layout(pad=0.1)

def main():
  parser = argparse.ArgumentParser(description='Estimate encoder nonlinearities')
  parser.add_argument('--input', required=True, type=str, help='path to CSV file containing measurements')
  parser.add_argument('--resolution', required=True, type=int, help='encoder resolution bits')
  parser.add_argument('--saveto', required=None, type=str, help='path to resulting json file')
  parser.add_argument('--plot-velocity', default=False, type=bool, help='plot velocity profile')
  parser.add_argument('--plot-result', default=False, type=bool, help='show calibration result')
  parser.add_argument('--nintervals', default=100, type=int, help='number of lookup table intervals')

  args = parser.parse_args()
  data = load_data(args.input)
  data = prepare_data(data, args.resolution)

  res = fit_lookup_table(data, args.nintervals)
  print('calibration result:')
  print(f'  fitting error: {res.fitting_error:.8f}')

  if args.saveto is not None:
    save_encoder_parameters(args.saveto, res)

  plot = False
  if args.plot_result:
    plot = True
    plt.figure('lookup table')
    plot_fitted(res)

  if args.plot_velocity:
    plot = True
    plt.figure('velocity')
    plot_velocity(data, res)

  if plot:
    plt.show()

if __name__ == '__main__':
  main()
