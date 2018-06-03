import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def ave(array):
    a_x = 0.0
    a_y = 0.0
    a_z = 0.0
    for i in array:
        a_x += i[0] / 30
        a_y += i[1] / 30
        a_z += i[2] / 30

    return a_x, a_y, a_z

def covariance_matrix(array, aveX, aveY, aveZ):
    V = np.zeros((3, 3))
    for i in array:
        V[0][0] += ((i[0] - aveX)**2) / 30
        V[1][1] += ((i[1] - aveY)**2) / 30
        V[2][2] += ((i[2] - aveZ)**2) / 30
        V[0][1] += ((i[0] - aveX)*(i[1] - aveY)) / 30
        V[0][2] += ((i[0] - aveX)*(i[2] - aveZ)) / 30
        V[1][2] += ((i[1] - aveY)*(i[2] - aveZ)) / 30

    V[1][0] = V[0][1]
    V[2][0] = V[0][2]
    V[2][1] = V[1][2]
    return V

def normalization(array, aveX, aveY, aveZ):
    n_array = np.zeros_like(array)

    n_array[:,0] = array[:,0] - aveX
    n_array[:,1] = array[:,1] - aveY
    n_array[:,2] = array[:,2] - aveZ

    return n_array



csv_input = pd.read_csv(filepath_or_buffer="6-2.csv", sep=",", engine="python")

array = csv_input.values
aveX, aveY, aveZ = ave(array)
V = covariance_matrix(array, aveX, aveY, aveZ)
lam, pro_v = np.linalg.eig(V)

n_array = np.zeros_like(array)
n_array = normalization(array, aveX, aveY, aveZ)

print(array)
print("japanese average:", aveX)
print("math average:", aveY)
print("english average:", aveZ)
print("V: ")
print(V)
print("eigen_value:")
print(lam)
print("proper_vector:")
print(pro_v)

X = np.arange(-30, 30, 1)

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(n_array[:,0], n_array[:,1], n_array[:,2], label='score')
plt.plot(X*pro_v[0][0], X*pro_v[1][0], X*pro_v[2][0], label="u1")
plt.plot(X*pro_v[0][1], X*pro_v[1][1], X*pro_v[2][1], label="u2")
plt.plot(X*pro_v[0][2], X*pro_v[1][2], X*pro_v[2][2], label="u3")
plt.legend()
ax1.set_xlabel('japanese')
ax1.set_ylabel('math')
ax1.set_zlabel('english')
plt.show()
