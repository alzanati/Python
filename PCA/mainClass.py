"""
@author:
@Date: 23 / 4 / 2016
@purpose: Implement Principle Component Analysis in python
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class PCA:
    def __init__(self, path):
        self.raw_data = self.loadData(path)
        self.feature_vector = np.zeros(0)
        self.cov_matrix = np.zeros(0)

    # load data from csv file
    def loadData(self, path):
        df = pd.read_csv(filepath_or_buffer=path, header=None, sep=',')
        df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        df.dropna(how="all", inplace=True)  # drops the empty line at file-end
        return df

    def showRawData(self):
        print(type(self.raw_data), '\n', self.raw_data)
        # print('\nLast Five Records\n', self.raw_data.tail())

    # calculate covariance matrix
    @staticmethod
    def covariance(data):
        mean = np.mean(data, axis=0)  # mean of data
        zero_mean_data = data - mean  # zero mean data
        data_transposed = np.transpose(zero_mean_data)  # transpose data
        cov_matrix = np.dot(data_transposed, zero_mean_data)  # matrix multiplication X * X'
        cov_matrix /= (data.shape[0] - 1)  # cov / n - 1
        return cov_matrix

    @staticmethod
    def normalization(vector):
        vector = np.multiply(vector, vector)
        return np.sqrt(sum(vector))

    # create feature vectors
    def createFeatureVector(self, start, end):
        self.feature_vector = self.raw_data.ix[:, start:end].values
        return self.feature_vector

    # show features vector
    def showFeatureVecotr(self):
        print(type(self.feature_vector), '\n', self.feature_vector)

    # return standrad data with mean = 0 and variance of 1
    def getStandardData(self):
        X_std = StandardScaler().fit_transform(self.feature_vector)
        return X_std

    def plotData(self, data, numberOfClass = 2):
        rows, cols = data.shape
        for col in range(cols-1):
            plt.plot(data[0:rows / 2, col], data[0:rows / 2, col+1], 'o',
                     markersize=7, color='blue', alpha=0.5,
                     label='class1')
            plt.plot(data[rows / 2:rows, col], data[rows / 2:rows, col+1], '^',
                     markersize=7, color='red', alpha=0.5,
                     label='class2')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples with class labels')

        plt.show()

if __name__ == '__main__':

    # 1 load data >> get raw data
    pca = PCA(path="/home/mohamed/work_space/python/PCA/iris.data.csv")

    # 2 create feature vector >> create 150 x 4 features vector
    featurs = pca.createFeatureVector(0, 4)

    # 3 calculate covariance matrix
    cov_matrix = PCA.covariance(featurs)

    # 4 calculate eigen vectors and values
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    print('eigen values = \n', eig_vals)
    print('\neigen vectors = \n', eig_vecs)

    # 5 select max eigen vectors with
    for ev in eig_vecs:
        print(PCA.normalization(ev))
        # assert normalized value are returned
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')

    # 6 Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()

    # 7 compute w matrix >>  4 x 2
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                          eig_pairs[1][1].reshape(4, 1)))

    # 8 new data space >> (150 x 4) * (4 x 2) = (150 x 2)
    newSpace = pca.getStandardData().dot(matrix_w)

    # 9 show data
    pca.plotData(newSpace)

