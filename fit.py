"""
Author: Alex Stennnet
File: fit.py
Description:
    This file is intended to allow for an easier use of the ransac
    fitting algorithm
File restriction:
    must be of the format such that numpy.loadtxt gets values (x, y, z) for
    each point in the point cloud
"""

import argparse
from ransac import ransac
from PolynomialLeastSquaresModel import PolynomialLeastSquaresModel

parser = argparse.ArgumentParser(description="Segments ground from a LIDAR point cloud file")
parser.add_argument('input_file', help="name of file to segment from")
parser.add_argument('output_file', help="name of file output segmentation to")

args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

data = np.loadtxt(input_file)

model = PolynomialLeastSquaresModel([0,1], [2], 2)

ransac_fit, ransac_data = ransac(data,model,
                                 n=20,
                                 k=500,
                                 t=.1,
                                 d=7000,
                                 m=200,
                                 return_all=True)

data = np.concatenate((x[ransac_data['inliers']], np.array([y[ransac_data['inliers']]]).T), axis=1)
np.savetxt(output_file, data[ransac_data['inliers']])
