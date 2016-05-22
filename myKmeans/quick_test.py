"""

"""

import numpy as np


# return a column features vector
def get_raw_data(x,y):
    z = np.stack((x, y))
    z = z.reshape((-1, 2))
    return z

def get_rand_centers(k, data):
    size = len(data)
    init_centers = []
    for i in range(k):
        index = np.random.randint(0, size)
        init_centers.append(data[index])
    return init_centers


if __name__ == '__main__':
    x = np.random.randint(0, 25, 10)
    y = np.random.randint(0, 25, 10)
    z = np.stack((x, y))
    z = z.reshape(-1, 2)
    c = get_rand_centers(3, z)
    print(c)