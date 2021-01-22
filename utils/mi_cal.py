# -*- coding: utf-8 -*-
import sys 
sys.path.append('../')
from config import *
import pandas as pd 
import numpy as np

def MI_cal(label_matrix,layer_T):
	'''
	Inputs:  
	- size_of_test: (N,) how many test samples have be given. since every input is different
	  we only care the number.
	-  label_matrix: (N,C)  the label_matrix created by creat_label_matrix.py.
	-  layer_T:  (N,H) H is the size of hidden layer
	Outputs:
	- MI_XT : the mutual information I(X,T)
	- MI_TY : the mutual information I(T,Y)
	'''
	MI_XT=0
	MI_TY=0
	hidden_size = layer_T.shape[1]
	layer_T = Discretize(layer_T)
	XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK))
	Non_repeat=[]
	mark_list=[]
	for i in range(NUM_TEST_MASK):
		pre_mark_size = len(mark_list)
		if i==0:
			Non_repeat.append(i)
			mark_list.append(i)
			XT_matrix[i,i]=1
		else:
			for j in range(len(Non_repeat)):
				if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
					mark_list.append(Non_repeat[j])
					XT_matrix[i,Non_repeat[j]]=1
					break
		if pre_mark_size == len(mark_list):
			Non_repeat.append(Non_repeat[-1]+1)
			mark_list.append(Non_repeat[-1])
			XT_matrix[i,Non_repeat[-1]]=1
	
	XT_matrix = np.delete(XT_matrix,range(len(Non_repeat),NUM_TEST_MASK),axis=1)				
	P_layer_T = np.sum(XT_matrix,axis=0)/float(NUM_TEST_MASK)
	P_sample_x = 1/float(NUM_TEST_MASK)
	for i in range(NUM_TEST_MASK):
		MI_XT+=P_sample_x*np.log2(1.0/P_layer_T[mark_list[i]])


	TY_matrix = np.zeros((len(Non_repeat),NUM_LABEL))
	mark_list = np.array(mark_list)
	for i in range(len(Non_repeat)):
		TY_matrix[i,:] = np.sum(label_matrix[  np.where(mark_list==i)[0]  , : ] ,axis=0 )
	TY_matrix = TY_matrix/NUM_TEST_MASK
	P_layer_T_for_Y = np.sum(TY_matrix,axis=1)
	P_Y_for_layer_T = np.sum(TY_matrix,axis=0)
	for i in range(TY_matrix.shape[0]):
		for j in range(TY_matrix.shape[1]):
			if TY_matrix[i,j]==0:
				pass
			else:
				MI_TY+=TY_matrix[i,j]*np.log2(TY_matrix[i,j]/(P_layer_T_for_Y[i]*P_Y_for_layer_T[j]))
	
	return MI_XT,MI_TY

def Discretize(layer_T):
	'''
	Discretize the output of the neuron 
	Inputs:
	- layer_T:(N,H)
	Outputs:
	- layer_T:(N,H) the new layer_T after discretized
	'''
	labels = np.arange(NUM_INTERVALS)
	pos_list = np.arange(NUM_INTERVALS/2+1)*(1.0/(NUM_INTERVALS/2))
	neg_list = -pos_list
	neg_list.sort()
	bins = np.append(neg_list,pos_list)
	bins = np.delete(bins,NUM_INTERVALS/2)
	for i in range(layer_T.shape[1]):
		temp = pd.cut(layer_T[:,i],bins,labels=labels)
		layer_T[:,i] = np.array(temp)
	return layer_T



def MI_cal_v2(label_matrix, layer_T, NUM_TEST_MASK):
	'''
	Inputs:  
	- size_of_test: (N,) how many test samples have be given. since every input is different
	  we only care the number.
	-  label_matrix: (N,C)  the label_matrix created by creat_label_matrix.py.
	-  layer_T:  (N,H) H is the size of hidden layer
	Outputs:
	- MI_XT : the mutual information I(X,T)
	- MI_TY : the mutual information I(T,Y)
	'''
	MI_XT=0
	MI_TY=0
	#hidden_size = layer_T.shape[1]



	#layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True))
	#layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	#layer_T = layer_T - np.min(layer_T,axis=1,keepdims=True)
	#layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	#column_add = hidden_size/NUM_INTERVALS
	#new_layer_T = []
	#for i in range(NUM_INTERVALS):
		#new_layer_T.append(np.sum(layer_T[:,column_add*i:column_add*i+column_add],axis=1,keepdims=True))
	#layer_T = np.column_stack(new_layer_T)
	#print(layer_T.shape)
	#layer_T = layer_T - np.min(layer_T,axis=1,keepdims=True)
	#layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True))
	layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	layer_T = Discretize_v2(layer_T)
	XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK))
	Non_repeat=[]
	mark_list=[]
	for i in range(NUM_TEST_MASK):
		pre_mark_size = len(mark_list)
		if i==0:
			Non_repeat.append(i)
			mark_list.append(i)
			XT_matrix[i,i]=1
		else:
			for j in range(len(Non_repeat)):
				if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
					mark_list.append(Non_repeat[j])
					XT_matrix[i,Non_repeat[j]]=1
					break
		if pre_mark_size == len(mark_list):
			Non_repeat.append(Non_repeat[-1]+1)
			mark_list.append(Non_repeat[-1])
			XT_matrix[i,Non_repeat[-1]]=1
	
	XT_matrix = np.delete(XT_matrix,range(len(Non_repeat),NUM_TEST_MASK),axis=1)				
	P_layer_T = np.sum(XT_matrix,axis=0)/float(NUM_TEST_MASK)
	P_sample_x = 1/float(NUM_TEST_MASK)
	for i in range(NUM_TEST_MASK):
		MI_XT+=P_sample_x*np.log2(1.0/P_layer_T[mark_list[i]])


	TY_matrix = np.zeros((len(Non_repeat),NUM_LABEL))
	mark_list = np.array(mark_list)
	for i in range(len(Non_repeat)):
		TY_matrix[i,:] = np.sum(label_matrix[  np.where(mark_list==i)[0]  , : ] ,axis=0 )
	TY_matrix = TY_matrix/NUM_TEST_MASK
	P_layer_T_for_Y = np.sum(TY_matrix,axis=1)
	P_Y_for_layer_T = np.sum(TY_matrix,axis=0)
	for i in range(TY_matrix.shape[0]):
		for j in range(TY_matrix.shape[1]):
			if TY_matrix[i,j]==0:
				pass
			else:
				MI_TY+=TY_matrix[i,j]*np.log2(TY_matrix[i,j]/(P_layer_T_for_Y[i]*P_Y_for_layer_T[j]))
	
	return MI_XT,MI_TY

def Discretize_v2(layer_T):
	'''
	Discretize the output of the neuron 
	Inputs:
	- layer_T:(N,H)
	Outputs:
	- layer_T:(N,H) the new layer_T after discretized
	'''
	'''
	interval_size = layer_T.shape[1]
	labels = np.arange(interval_size)
	bins = np.arange(interval_size+1)
	bins = bins/float(interval_size)
	'''
	
	labels = np.arange(NUM_INTERVALS)
	bins = np.arange(NUM_INTERVALS+1)
	bins = bins/float(NUM_INTERVALS)
	
	for i in range(layer_T.shape[1]):
		temp = pd.cut(layer_T[:,i],bins,labels=labels)
		layer_T[:,i] = np.array(temp)
	return layer_T


def MI_cal_v3(label_matrix, layer_T, NUM_TEST_MASK, NUM_LABEL):
	'''
	Inputs:  
	- size_of_test: (N,) how many test samples have be given. since every input is different
	  we only care the number.
	-  label_matrix: (N,C)  the label_matrix created by creat_label_matrix.py.
	-  layer_T:  (N,H) H is the size of hidden layer
	Outputs:
	- MI_XT : the mutual information I(X,T)
	- MI_TY : the mutual information I(T,Y)
	'''
	MI_XT=0
	MI_TY=0
	hidden_size = layer_T.shape[1]
	layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True))
	layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	#layer_T = layer_T - np.min(layer_T,axis=1,keepdims=True)
	#layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	#column_add = hidden_size/NUM_INTERVALS
	#new_layer_T = []
	#for i in range(NUM_INTERVALS):
		#new_layer_T.append(np.sum(layer_T[:,column_add*i:column_add*i+column_add],axis=1,keepdims=True))
	#layer_T = np.column_stack(new_layer_T)
	#print layer_T.shape
	#layer_T = layer_T - np.min(layer_T,axis=1,keepdims=True)
	#layer_T /= np.sum( layer_T,axis=1,keepdims=True)
	#print layer_T[0]
	#layer_T /= (np.max( layer_T,axis=1,keepdims=True)+1e-3)
	#print layer_T[0]
	layer_T = Discretize_v3(layer_T)
	#print layer_T[0]
	XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK))
	Non_repeat=[]
	mark_list=[]
	for i in range(NUM_TEST_MASK):
		pre_mark_size = len(mark_list)
		if i==0:
			Non_repeat.append(i)
			mark_list.append(i)
			XT_matrix[i,i]=1
		else:
			for j in range(len(Non_repeat)):
				if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
					mark_list.append(Non_repeat[j])
					XT_matrix[i,Non_repeat[j]]=1
					break
		if pre_mark_size == len(mark_list):
			Non_repeat.append(Non_repeat[-1]+1)
			mark_list.append(Non_repeat[-1])
			XT_matrix[i,Non_repeat[-1]]=1
	
	XT_matrix = np.delete(XT_matrix,range(len(Non_repeat),NUM_TEST_MASK),axis=1)				
	P_layer_T = np.sum(XT_matrix,axis=0)/float(NUM_TEST_MASK)
	P_sample_x = 1/float(NUM_TEST_MASK)
	for i in range(NUM_TEST_MASK):
		MI_XT+=P_sample_x*np.log2(1.0/P_layer_T[mark_list[i]])


	TY_matrix = np.zeros((len(Non_repeat),NUM_LABEL))
	mark_list = np.array(mark_list)
	for i in range(len(Non_repeat)):
		TY_matrix[i,:] = np.sum(label_matrix[  np.where(mark_list==i)[0]  , : ] ,axis=0 )
	TY_matrix = TY_matrix/NUM_TEST_MASK
	P_layer_T_for_Y = np.sum(TY_matrix,axis=1)
	P_Y_for_layer_T = np.sum(TY_matrix,axis=0)
	for i in range(TY_matrix.shape[0]):
		for j in range(TY_matrix.shape[1]):
			if TY_matrix[i,j]==0:
				pass
			else:
				MI_TY+=TY_matrix[i,j]*np.log2(TY_matrix[i,j]/(P_layer_T_for_Y[i]*P_Y_for_layer_T[j]))
	
	return MI_XT,MI_TY

def Discretize_v3(layer_T):
	'''
	Discretize the output of the neuron 
	Inputs:
	- layer_T:(N,H)
	Outputs:
	- layer_T:(N,H) the new layer_T after discretized
	'''
	'''
	interval_size = layer_T.shape[1]
	labels = np.arange(interval_size)
	bins = np.arange(interval_size+1)
	bins = bins/float(interval_size)
	'''
	
	labels = np.arange(2)
	bins = np.arange(2+1)
	bins = bins/float(2)
	#bins[0]=bins[0]-1e-3
	for i in range(layer_T.shape[1]):
		temp = pd.cut(layer_T[:,i],bins,labels=labels)
		layer_T[:,i] = np.array(temp)
	return layer_T







'''
image_test = np.load('/home/chenghao/桌面/IB_NN/Data/MNIST/image_test.npy')
image_test = image_test/255.0
label_test = np.load( '/home/chenghao/桌面/IB_NN/Data/MNIST/label_test.npy')
label_matrix = np.load('/home/chenghao/桌面/IB_NN/Data/MNIST/label_test_matrix.npy')
image_test = image_test[0:NUM_TEST_MASK]
label_test = label_test[0:NUM_TEST_MASK]
label_matrix = label_matrix[0:NUM_TEST_MASK]
image_test = np.reshape(image_test, (image_test.shape[0],-1) )
W = np.random.normal(0,1,(28*28,3))
T_Value = image_test.dot(W)
T_Value_q = 1.0/(1+np.exp(-T_Value))
xt,ty = MI_cal(label_matrix,T_Value_q)
print xt,ty



W2 = np.random.normal(0,1,(3,4))
T_Value2 = T_Value_q.dot(W2)
T_Value2_q = 1.0/(1+np.exp(-T_Value2))
xt,ty = MI_cal(label_matrix,T_Value2_q)

print xt,ty

'''