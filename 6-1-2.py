import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normalization(array, aveX, aveY):
    array[:,0] = array[:,0] - aveX
    array[:,1] = array[:,1] - aveY

    return array

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

n_array = normalization(array, aveX, aveY)

x = np.arange(-100, 100, 1)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-30, 30])
plt.ylim([-30, 30])
plt.plot(n_array[:,0], n_array[:,1], "o")
plt.plot(x*pro_v[0][0], x*pro_v[1][0], linestyle="dashed")
plt.plot(x*pro_v[0][1], x*pro_v[1][1], linestyle="dashed")

plt.show()
