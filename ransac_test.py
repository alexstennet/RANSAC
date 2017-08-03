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

velo = next(velos)
velo = next(velos)
print(len(velo))

# velo = velo[velo[:,3] > .05]

# velo = np.loadtxt('all.txt')

velo = np.delete(velo, 3, axis=1)

x = np.delete(velo, 2, axis=1)
y = velo[:,2]

# model = ransac.LinearLeastSquaresModel([0,1], [2])
model = PolynomialLeastSquaresModel([0,1], [2], 2)

ransac_fit, ransac_data = ransac(velo,model,
                                 n=20, k=500, t=.1, d=7000, m=200, # misc. parameters
                                 return_all=True)
data = np.concatenate((x[ransac_data['inliers']], np.array([y[ransac_data['inliers']]]).T), axis=1)
np.savetxt('./' + 'blah' + drive + '.txt', velo[ransac_data['inliers']])
