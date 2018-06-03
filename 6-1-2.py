import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normalization(array, aveX, aveY):
    n_array = array
    n_array[:,0] = array[:,0] - aveX
    n_array[:,1] = array[:,1] - aveY



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


normalization(array, aveX, aveY)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.plot(array[:,0], array[:,1], "o")
plt.plot(array[:,0]*pro_v[0][0], array[:,0]*pro_v[1][0], color='r', linestyle="solid", label="ξa")
plt.plot(array[:,0]*pro_v[0][1], array[:,0]*pro_v[1][1], color='g', linestyle="solid", label="ηa")

plt.legend()
plt.show()
