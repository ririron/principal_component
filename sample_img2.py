import numpy as np
import matplotlib.pylab as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):

    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def ave_image(array):
    ave = np.zeros((28, 28))
    s = 0
    for i in range(60000):
        img = array[i].reshape(28, 28)
        ave += img

    ave = ave / i
    plt.imshow(ave)
    plt.show()

    return ave

def diff_ave(x_train, diff_a, ave):
    diff_a = x_train - ave.reshape(784)
    return diff_a



(x_train, t_train),(x_test, t_test) = load_mnist(flatten = True, normalize = False)

#print(x_train.shape)

#img_show(x_train[0])

ave = ave_image(x_train)
diff = np.zeros_like(x_train)
diff = diff_ave(x_train, diff, ave)
plt.imshow(x_train[0].reshape(28,28))
plt.show()
plt.imshow(diff[0].reshape(28,28))
plt.show()
