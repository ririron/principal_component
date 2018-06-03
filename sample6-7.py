import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def normalization(array, aveX, aveY):
    n_array = np.zeros((array.shape[0],array.shape[1]))

    n_array[:,0] = array[:,0] - aveX
    n_array[:,1] = array[:,1] - aveY

    return n_array


def ave(array):
    a_x = 0.0
    a_y = 0.0
    for i in array:
        a_x += i[0] / 30
        a_y += i[1] / 30

    return a_x, a_y

def covariance_matrix(array, aveX, aveY):
    V = np.zeros((2, 2))
    for i in array:
        V[0][0] += ((i[0] - aveX)**2) / 30
        V[0][1] += ((i[0] - aveX)*(i[1] - aveY)) / 30
        V[1][0] += ((i[0] - aveX)*(i[1] - aveY)) / 30
        V[1][1] += ((i[1] - aveY)**2) / 30

    return V


csv_input = pd.read_csv(filepath_or_buffer="6-1.csv", sep=",", engine="python")

array = csv_input.values

aveX, aveY = ave(array)
print(array)
print("height ave:", aveX)
print("weight ave:", aveY)

V = covariance_matrix(array, aveX, aveY)
print("V:", V)

lam, pro_v = np.linalg.eig(V)

print("eigen_value:", lam)
print("proper_vector:", pro_v)

n_array = np.zeros((array.shape[0],array.shape[1]))
n_array = normalization(array, aveX, aveY)

X = np.arange(-30, 30, 1)

fig = plt.figure(figsize=(25, 10))
ax1 = fig.add_subplot(1, 3, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-20, 20])
plt.ylim([-20, 20])
ax1.plot(n_array[:,0], n_array[:,1], "o")
ax1.plot(X*pro_v[0][0], X*pro_v[1][0], color='r', linestyle="solid", label="u1")
ax1.plot(X*pro_v[0][1], X*pro_v[1][1], color='g', linestyle="solid", label="u2")

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(array[:,0], array[:,1], pro_v[0][0]*n_array[:,0]+pro_v[1][0]*n_array[:,1], c='b', marker='o', label='ξ')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(array[:,0], array[:,1], pro_v[0][1]*n_array[:,0]+pro_v[1][1]*n_array[:,1], c='r', marker='^', label='η')

plt.legend()
plt.show()
