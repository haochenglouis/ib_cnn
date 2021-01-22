# -*- coding:utf-8 -*-

# Import some library
import sys 
sys.path.append('../')
from config import *
import numpy as np 
import random

# Load raw mnist data(npy files)

image_train = np.load(train_image_path)
label_train = np.load(train_label_path)
image_test = np.load(test_image_path)
label_test = np.load(test_label_path)
label_test_matrix = np.load(test_label_matrix_path)

# Creat mnist class

class mnist_input(object):
    def __init__(self):
        self.train_data = np.reshape(image_train[0:50000],(50000,-1))/255.0
        self.train_label = label_train[0:50000]
        self.eval_data =np.reshape(image_train[50000:60000],(10000,-1))/255.0
        self.eval_label = label_train[50000:60000]
        self.test_data = np.reshape(image_test,(10000,-1))/255.0
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = BATCH_SIZE):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        return train_image, train_label
    def eval_total(self):
        return self.eval_data,self.eval_label
    def test(self,mask=NUM_TEST_MASK):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]

class iris_input(object):
    def __init__(self):
        self.train_data = np.reshape(image_train[0:80],(80,-1))
        self.train_label = label_train[0:80]
        self.eval_data =np.reshape(image_train[80:90],(10,-1))
        self.eval_label = label_train[80:90]
        self.test_data = np.reshape(image_test,(60,-1))
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = 6):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        return train_image, train_label
    def eval_total(self):
        return self.eval_data,self.eval_label
    def test(self,mask):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]



class wine_input(object):
    def __init__(self):
        self.train_data = np.reshape(image_train[0:100],(100,-1))
        self.train_label = label_train[0:100]
        self.eval_data =np.reshape(image_train[100:120],(20,-1))
        self.eval_label = label_train[100:120]
        self.test_data = np.reshape(image_test,(58,-1))
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = 20):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        return train_image, train_label
    def eval_total(self):
        return self.eval_data,self.eval_label
    def test(self,mask):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]


class cifar10_input(object):
    def __init__(self):
        self.train_data = np.reshape(image_train[0:49000],(49000,-1))
        self.train_label = label_train[0:49000]
        self.eval_data =np.reshape(image_train[49000:50000],(1000,-1))
        self.eval_label = label_train[49000:50000]
        self.test_data = np.reshape(image_test,(10000,-1))
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = BATCH_SIZE):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        return train_image, train_label
    def eval_total(self):
        return self.eval_data,self.eval_label
    def test(self,mask=NUM_TEST_MASK):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]



class cifar10_cnn_input(object):
    def __init__(self):
        self.train_data = image_train[0:49000]
        self.train_label = label_train[0:49000]
        self.eval_data =image_train[49000:50000]
        self.eval_label = label_train[49000:50000]
        self.test_data = image_test
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = BATCH_SIZE):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        return train_image, train_label
    def eval_total(self):
        return self.eval_data,self.eval_label
    def test(self,mask=NUM_TEST_MASK):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]


class cifar100_cnn_input(object):
    def __init__(self):
        self.train_data = image_train[0:49000]
        self.train_label = label_train[0:49000]
        self.eval_data =image_train[49000:50000]
        self.eval_label = label_train[49000:50000]
        self.test_data = image_test
        self.test_label = label_test
        self.label_test_matrix = label_test_matrix
    def train_batch(self,batchsize = BATCH_SIZE):
        num = range(self.train_data.shape[0])
        sample_num = random.sample(num,batchsize)
        train_image = self.train_data[sample_num]
        train_label = self.train_label[sample_num]
        #label_train_matrix = np.zeros((BATCH_SIZE,NUM_LABEL))
        #label_train_matrix[np.arange(BATCH_SIZE) , train_label ] = 1



        return train_image, train_label
    def eval_total(self):
        #label_matrix = np.zeros((1000,NUM_LABEL))
        #label_matrix[np.arange(1000),self.eval_label] = 1
        return self.eval_data,self.eval_label
    def test(self,mask=NUM_TEST_MASK):
        return self.test_data[0:mask],self.test_label[0:mask],self.label_test_matrix[0:mask]
