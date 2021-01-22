# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')
from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

acc_list = []
xt_value = np.load('Baseline/mi_xt.npy')
ty_value = np.load('Baseline/mi_ty.npy')
accuracy = np.load('Baseline/val_acc.npy')
acc_list.append(accuracy[-1])
print('Baseline')
print(xt_value)
print(ty_value)
print(accuracy)


xt_value_2 = []
ty_value_2 = []
#xt_value.shape
for i in range(xt_value.shape[0]):
	if i % 2 == 1:
		xt_value_2.append(xt_value[i])
		ty_value_2.append(ty_value[i])


plt.plot(xt_value_2,ty_value_2, 'b', label='Baseline')
#plt.show()
#plt.savefig('Baseline/cifar10.png')


xt_value = np.load('Alexnet/mi_xt.npy')
ty_value = np.load('Alexnet/mi_ty.npy')
accuracy = np.load('Alexnet/val_acc.npy')
acc_list.append(accuracy[-1])
print('Alexnet')
print(xt_value)
print(ty_value)
print(accuracy)

xt_value_2 = []
ty_value_2 = []
#xt_value.shape
for i in range(xt_value.shape[0]):
	if i % 2 == 1:
		xt_value_2.append(xt_value[i])
		ty_value_2.append(ty_value[i])

plt.plot(xt_value_2,ty_value_2, 'g', label='Alexnet')


xt_value = np.load('Vgg/mi_xt.npy')
ty_value = np.load('Vgg/mi_ty.npy')
accuracy = np.load('Vgg/val_acc.npy')
acc_list.append(accuracy[-1])
print('Vgg')
print(xt_value)
print(ty_value)
print(accuracy)

xt_value_2 = []
ty_value_2 = []
#xt_value.shape
for i in range(xt_value.shape[0]):
	if i % 2 == 1:
		xt_value_2.append(xt_value[i])
		ty_value_2.append(ty_value[i])


plt.plot(xt_value_2,ty_value_2, 'r', label='Vgg')


xt_value = np.load('Resnet/mi_xt.npy')
ty_value = np.load('Resnet/mi_ty.npy')
accuracy = np.load('Resnet/val_acc.npy')
acc_list.append(accuracy[-1])
print('Resnet')
print(xt_value)
print(ty_value)
print(accuracy)

xt_value_2 = []
ty_value_2 = []
#xt_value.shape
for i in range(xt_value.shape[0]):
	if i % 2 == 1:
		xt_value_2.append(xt_value[i])
		ty_value_2.append(ty_value[i])

plt.plot(xt_value_2,ty_value_2, 'k', label='Resnet')




xt_value = np.load('Densenet/mi_xt.npy')
ty_value = np.load('Densenet/mi_ty.npy')
accuracy = np.load('Densenet/val_acc.npy')
acc_list.append(accuracy[-1])
print('Densenet')
print(xt_value)
print(ty_value)
print(accuracy)
xt_value_2 = []
ty_value_2 = []
#xt_value.shape
for i in range(xt_value.shape[0]):
	if i % 2 == 1:
		xt_value_2.append(xt_value[i])
		ty_value_2.append(ty_value[i])


print('acc_list: {}'.format(acc_list))

plt.plot(xt_value_2,ty_value_2, 'y', label='Densenet')
plt.xlabel('I(X;T)')
plt.ylabel('I(T;Y)')
plt.title('Relation between mutual information')
plt.legend()
#plt.show()
plt.savefig('cifar10.png')
