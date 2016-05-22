"""
@author: Mohamed Hosny Ahmed
@purpose: test kmeans implementation
"""

import numpy as np
import cv2


#   data: [[2,3]\n[2,5],[5,8]]
class kmeans:
    def __init__(self, data, k=3, iterations=10):
        self.data = data
        self.reduced_data = np.array(0)
        self.size = len(self.data)  # get length of data number of points
        self.features = len(data[0])
        self.k = k
        self.max_iter = iterations
        self.itr = 0
        self.labels_array = np.array(0)
        self.labels = []
        self.lables_names = []
        self.old_centers = []
        self.new_centers = []
        self.clusters = []
        self.centroids = []

    # get list of random centers to start algorithm
    def get_rand_centers(self):
        old = np.array([0, 0])
        for i in range(self.k):
            index = np.random.randint(0, self.size)
            new_center = self.data[index]
            while True:
                if new_center[0] != old[0] or new_center[1] != old[1]:
                    self.old_centers.append(new_center)
                    old = new_center
                    break
                else:
                    index = np.random.randint(0, self.size)
                    new_center = self.data[index]
        return self.old_centers

    # create a label name  to each center in order
    def get_lables_names(self):
        for i in range(self.k):
            self.lables_names.append(i)
        self.lables_names = np.array(self.lables_names).reshape(1, -1)
        return self.lables_names

    # return lables to each point
    def get_lables(self):
        itr = 0
        self.clusters = [[] for i in range(self.k)]
        self.new_centers = self.old_centers

        for center in self.new_centers:
            tmp = self.get_distance_center_vector(center)
            self.clusters[itr].append(tmp)  # label vector of k columns with all distance
            itr += 1

        cluster_array = np.array(self.clusters)
        cluster_array = cluster_array.T

        for i in range(len(cluster_array)):
            self.labels.append(cluster_array[i].argmin())

        self.labels = np.array(self.labels).reshape(1, -1)
        return self.labels

    # get clusters [[],[],[],....]
    def get_clusters(self):
        self.clusters = [[] for i in range(self.k)]
        lab = self.labels  # index of index of array
        lab = np.array(lab[0])
        names = np.array(self.lables_names[0])
        for i in range(self.size):
            for j in range(self.k):
                if lab[i] == names[j]:
                    self.clusters[lab[i]].append(self.data[i])
        return self.clusters

    # should give euclidean distance between one center and all points
    def get_distance_center_vector(self, center):
        vec = []
        self.reduced_data = (self.data - center) ** 2
        for i in range(len(self.reduced_data)):  # sum ((x - x1)^2 + (y - y1)^2)^0.5
            tmp = self.reduced_data[i]
            tmp = np.sqrt(sum(tmp))
            vec.append(tmp)
        return vec

    # get center of data according to sum points according to positions
    def get_mean_point(self):
        length = len(self.clusters)
        points = [[] for it in range(self.features)]
        t = []

        for i in range(length):
            tmp = self.clusters[i]
            for j in range(len(tmp)):  # point
                point = tmp[j]
                for k in range(len(point)):
                    points[k].append(point[k])

            for it in range(len(points)):
                pon = points[it]
                t.append(np.mean(pon))

            self.centroids.append(t)
            t = []
            points = [[] for it in range(self.features)]

        self.new_centers = self.centroids
        return self.centroids

    # run algorithm with set of iterations
    def run_algorithm(self):

        self.get_rand_centers()
        self.get_lables_names()

        for i in range(self.max_iter):
            self.itr += 1
            labels = self.get_lables()
            self.get_clusters()
            centers = k.get_mean_point()

            self.clusters = []
            self.centroids = []
            self.labels = []

        return np.int32(labels), np.int32(centers)


if __name__ == '__main__':
    img = cv2.imread('/home/prof/Pictures/seg3.png')
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    k = kmeans(k=3, data=Z, iterations=10)
    label, center = k.run_algorithm()

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
