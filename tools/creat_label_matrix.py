# -*- coding: utf-8 -*-
#from config import *
import numpy as np 
import torch
import os
from torchvision import datasets





# Get MNIST label_test_matrix

test_label_matrix_path = '../label_test_matrix/mnist/label_test_matrix.npy'

root = '../data/mnist'
processed_folder = 'processed'
test_file = 'training.pt'

test_data, test_labels = torch.load(
                os.path.join(root, processed_folder, test_file))
test_labels = test_labels.numpy()
#print(test_labels.shape)
# label_test = np.load(test_label_path)
label_test_matrix = np.zeros((60000, 10))
label_test_matrix[np.arange(60000) , test_labels] = 1
#print(label_test_matrix[:4])
np.save(test_label_matrix_path, label_test_matrix)









# Get CIFAR10 label_test_matrix

test_label_matrix_path = '../label_test_matrix/cifar10/label_test_matrix.npy'
root = '../data/cifar10'
cifar10 = datasets.CIFAR10(root, train=False)
label_test_matrix = np.zeros((10000, 10))
label_test_matrix[np.arange(10000) , cifar10.test_labels] = 1
print(label_test_matrix[:4])
np.save(test_label_matrix_path, label_test_matrix)










# Get CIFAR100 label_test_matrix

#test_label_matrix_path = '../label_test_matrix/cifar100/label_test_matrix.npy'
#root = '../data/cifar100'
#cifar100 = datasets.CIFAR100(root, train=False)
#label_test_matrix = np.zeros((10000, 100))
#label_test_matrix[np.arange(10000) , cifar100.test_labels] = 1
#print(label_test_matrix[:4])
#np.save(test_label_matrix_path, label_test_matrix)
