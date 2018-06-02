from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Define the number of clusters and provide an array with the different centers
def get_clusters(X,n=3):
    # Number of clusters
    print(X)
    k = n
    # X coordinates of random centroids
    C_x = np.random.randint(0, 30, size=k)
    # Y coordinates of random centroids
    C_y = np.random.randint(0, 30, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    C_final = update_clusters(k,X,C,error_range=0.02)
    return C_final

def update_clusters(k, X, C, error_range = 0.01):

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    cluster_iter = 1
    # Loop will run till the error becomes zero
    while error > error_range:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
        print("======> Cluster Iteration "+str(cluster_iter)+"\n---------------------------")
        print(C)
        print("error = "+str(error))
        cluster_iter += 1
    return {'C':C,'clusters':clusters}

