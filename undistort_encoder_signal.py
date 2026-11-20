import argparse 
import numpy as np
from calibrate_encoder import (
  EncoderParameters,
  load_encoder_parameters,
  undistort_angle,
  unwrap_angle
)
from numpy.lib import recfunctions as rfn

def main():
  parser = argparse.ArgumentParser(description='Estimate encoder nonlinearities')
  parser.add_argument('--input', required=True, type=str, help='path to CSV file containing input signal')
  parser.add_argument('--config', required=True, type=str, help='encoder calibration config')
  parser.add_argument('--output', type=str, help='path to the resulting CSV file')

  args = parser.parse_args()
  # data = load_data(args.input)
  encpar = load_encoder_parameters(args.config)
  data = np.genfromtxt(args.input, delimiter=',', names=True)
  angle = data['angle']
  undistorted = undistort_angle(angle, encpar)
  
  if args.output is not None:
    new_data = rfn.append_fields(data, 'undistorted', undistorted, usemask=False)
    header_str = ",".join(new_data.dtype.names)
    np.savetxt(
      args.output,
      new_data,
      delimiter=", ",
      header=header_str,
      comments="",
    )
  else:
    print('undistorted:')
    print(undistorted)

if __name__ == '__main__':
  main()
