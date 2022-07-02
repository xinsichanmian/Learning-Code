from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

data_dir = '../dataset/mmist_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
mnist = input_data.read_data_sets(data_dir, one_hot=True)
mnist_train = (mnist.train.images.reshape(55000, 28, 28, 1) * 255).astype(np.uint8)
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)

# mnist_inv = mnist_train * (-1) + 255
# mnist_train = np.concatenate([mnist_train, mnist_inv])
mnist_test = (mnist.test.images.reshape(10000, 28, 28, 1) * 255).astype(np.uint8)
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
mnist_valid = (mnist.validation.images.reshape(5000, 28, 28, 1) * 255).astype(np.uint8)
mnist_valid = np.concatenate([mnist_valid, mnist_valid, mnist_valid], 3)
# dataset['mnist']['train'] = {'images': mnist_train, 'labels': np.concatenate([mnist.train.labels, mnist.train.labels])}
print(mnist_test.shape)
print(mnist)