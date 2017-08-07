from ransac import ransac
from PolynomialLeastSquaresModel import PolynomialLeastSquaresModel

import pykitti
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

basedir = "../KITTI Datasets/"
date = "2011_09_26"
drive = "0001"

velos = pykitti.raw(basedir, date, drive).velo


def get_velo(i, velos):
    velo = next(velos)
    while i > 0:
        velo = next(velos)
        i -= 1
    return velo


i = 0
velo = get_velo(i, velos)

# velo = velo[velo[:,3] > .05]

# velo = np.loadtxt('all.txt')

velo = np.delete(velo, 3, axis=1)

def fit(data, iterations, threshold):
    model = PolynomialLeastSquaresModel([0,1], [2], 2)
    ransac_fit, ransac_data = ransac(data,model,
                                     n=20,
                                     k=iterations,
                                     t=threshold, # threshold
                                     d=7000,
                                     m=200, #
                                     return_all=True)
    return velo[ransac_data['inliers']]

seg = fit(velo, 100, .25)
np.savetxt('./{}1.txt'.format(i), seg)

seg2 = fit(seg, 200, .1)
np.savetxt('./{}2.txt'.format(i), seg2)

seg3 = fit(seg2, 400, .075)
np.savetxt('./{}3.txt'.format(i), seg3)

seg4 = fit(seg3, 800, .05)
np.savetxt('./{}4.txt'.format(i), seg4)
#
