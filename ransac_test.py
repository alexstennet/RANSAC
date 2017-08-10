from ransac import ransac
from PolynomialLeastSquaresModel import PolynomialLeastSquaresModel
import os

import sys

import pykitti
import numpy as np

basedir = "../KITTI_raw/"
date = sys.argv[1] #"2011_09_26"
drive = sys.argv[2] #"0001"

velos = pykitti.raw(basedir, date, drive).velo

i=0
velo = next(velos)

# velo = velo[velo[:,3] > .05]

# velo = np.loadtxt('all.txt')

def fit(data, iterations, threshold):
    model = PolynomialLeastSquaresModel([0,1], [2], 2)
    ransac_fit, ransac_data = ransac(data,model,
                                     n=20,
                                     k=iterations,
                                     t=threshold, # threshold
                                     d=7000,
                                     m=200,
                                     return_all=True)
    return data[ransac_data['inliers']]

os.makedirs('../segmented/{}/{}'.format(date, drive), exist_ok=True)

for velo in velos:
    velo = np.delete(velo, 3, axis=1)

    seg = fit(velo, 200, .1)
    np.savetxt('../segmented/{}/{}/{}-1.txt'.format(date, drive, i), seg)

    seg2 = fit(seg, 200, .05)
    np.savetxt('../segmented/{}/{}/{}-2.txt'.format(date, drive, i), seg2)
    i += 1
#
# seg3 = fit(seg2, 400, .075)
# np.savetxt('./{}3.txt'.format(i), seg3)
#
# seg4 = fit(seg3, 800, .05)
# np.savetxt('./{}4.txt'.format(i), seg4)
