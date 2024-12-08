import numpy as np
from nn import MLP

from scipy.special import expit as activation_function

import matplotlib.pyplot as plt

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "./data/"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asarray(train_data[:, :1])
test_labels = np.asarray(test_data[:, :1])
lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(np.float64)
test_labels_one_hot = (lr == test_labels).astype(np.float64)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

simple_network = MLP(image_pixels, 10, 100, 0.1)
simple_network.train(train_imgs, train_labels_one_hot, 15, 0.9, 0.9, 10)



print(simple_network.get_classification_metrics(train_imgs, train_labels))
print(simple_network.get_classification_metrics(test_imgs, test_labels))
