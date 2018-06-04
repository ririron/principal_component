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

def ave_image(array, t_train):
    ave = np.zeros((28, 28))
    s = 0
    for i in range(60000):
        if t_train[i] == 4:
            img = array[i].reshape(28, 28)
            ave += img
            s += 1
    ave = ave / s
    plt.imshow(ave)
    plt.show()

    return ave




(x_train, t_train),(x_test, t_test) = load_mnist(flatten = True, normalize = False)

#print(x_train.shape)

#img_show(x_train[0])

ave = ave_image(x_train, t_train)
#diff_ave(x_train, t_train, ave)
