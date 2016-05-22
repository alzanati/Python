"""
@author: Mohamed Hos / np.float16(self.random)ny Ahmed
@purpose: implement wiener filter
"""

from randgenerator import RandomGenerator
import numpy as np
import matplotlib.pyplot as plt

class WienerFilter(RandomGenerator):
    def __init__(self, signalCount=10):
        RandomGenerator.__init__(self, start=1, end=100, size=10)
        self.signalCount = signalCount

    # calculate covariance matrix
    @staticmethod
    def covariance(data):
        mean = data.mean() # mean of data
        zero_mean_data = data - mean  # zero mean data
        data_transposed = np.transpose(zero_mean_data)  # transpose data
        cov_matrix = np.dot(data_transposed, zero_mean_data)  # matrix multiplication X * X'
        cov_matrix /= (data.shape[0] - 1)  # cov / n - 1
        print(cov_matrix.shape[0], 'x', cov_matrix.shape[1])
        return cov_matrix

    def get_random_data(self, size):
        self.rand_generator()
        return self.get_random_vector()

    # def get_noise_data(self, size):
    #
    # def run_filter(self):
    #
    # def

if __name__ == "__main__":
    fs = 100  # sample rate
    f = 4  # the frequency of the signal
    sigma = 3  # sigma for white noise
    j = 0

    signals = []
    noise = []
    output = []

    #   the points on the x axis for plotting
    x = np.arange(fs)
    theta = np.arange(start=-np.pi, step=(np.pi - (-np.pi)) / 100, stop=np.pi)  # uniform phase

    #   random time
    w = WienerFilter()
    t = w.get_random_data(size=10)
    t = t / max(t)

    #   generate random signals
    for j in np.arange(10):
        tmpSin = [(2 * np.sin(2 * np.pi * f * (t[j] / fs) + np.pi * theta[i])) for i in range(fs)]
        signals.append(tmpSin)
    signals = np.float32(signals)
    covX = WienerFilter.covariance(signals)

    #   generate random guassain noise
    t = np.random.randint(0, 100, size=100)
    t = t / max(t)
    for j in range(10):
        tmpL = []
        for i in range(100):
            tmpZ = (t[i] ** 2) / 2 * (sigma ** 2)
            tmp = (1 / 2 * np.sqrt(2 * np.pi)) * np.exp(-tmpZ)
            tmpL.append(tmp)
        noise.append(np.float32(tmpL))
    noise = np.float32(noise)
    covN = WienerFilter.covariance(noise)

    #   generate y = x + n
    for i in range(10):
        tmp = signals[i] + noise[i]
        output.append(tmp)
    output = np.float32(output)
    covY = WienerFilter.covariance(output)

    # new signals
    x11 = [(2 * np.sin(2 * np.pi * f * (i / fs) + np.pi * theta[i])) for i in range(fs)]
    n11 = []
    for i in range(100):
        tmpZ = (t[i] ** 2) / 2 * (sigma ** 2)
        tmp = (1 / 2 * np.sqrt(2 * np.pi)) * np.exp(-tmpZ)
        n11.append(tmp)
    n11 = np.float32(n11)
    y11 = n11 + x11

    # apply winer filter
    v = signals - output
    covXY = WienerFilter.covariance(v)
    A = covXY * np.linalg.inv(covY)
    b = signals.mean() - A * output.mean()
    estimated_x11 = A * y11 + b

    #   plot signals and cov matrix
    plt.figure()
    plt.imshow(covX)

    plt.figure()
    plt.imshow(covY)

    plt.figure()
    plt.imshow(covXY)

    plt.figure()
    plt.plot(x11)

    plt.figure()
    plt.plot(y11)

    plt.figure()
    plt.plot(estimated_x11)

    plt.show()