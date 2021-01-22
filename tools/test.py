# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')
#from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

acc_list = []
xt_value = []
ty_value = []

#plt.show()
#plt.savefig('Baseline/cifar10.png')




xt_value = np.load('mi_xt.npy')
ty_value = np.load('mi_ty.npy')
accuracy = np.load('val_acc.npy')
acc_list.append(accuracy[-1])
print('Vgg')
print(xt_value)
print(ty_value)
print(accuracy)

#xt_value.shape
# for i in range(xt_value.shape[0]):
# 	if i % 2 == 1:
# 		xt_value_2.append(xt_value[i])
# 		ty_value_2.append(ty_value[i])


plt.plot(xt_value,ty_value, 'r', label='Vgg')










print('acc_list: {}'.format(acc_list))


plt.xlabel('I(X;T)')
plt.ylabel('I(T;Y)')
plt.title('Relation between mutual information')
plt.legend()
#plt.show()
plt.savefig('cifar10.png')
