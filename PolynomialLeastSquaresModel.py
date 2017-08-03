from ransac import LinearLeastSquaresModel
from sklearn.preprocessing import PolynomialFeatures
import numpy, scipy

class PolynomialLeastSquaresModel:
    """docstring for PolynomialLeastSquaresModel."""
    def __init__(self, input_columns, output_columns, deg):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.poly = PolynomialFeatures(deg)
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        A = self.poly.fit_transform(A)
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error(self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        A = self.poly.fit_transform(A)
        B_fit = scipy.dot(A, model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
