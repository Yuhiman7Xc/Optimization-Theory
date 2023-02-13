# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:22:12 2019
@author: Ding
"""
import numpy as np
import matplotlib.pyplot as plt
import gzip as gz
#--------------------读取数据----------
#filename为文件名，kind为'data'或'lable'
def load_data(filename,kind):
    with gz.open(filename, 'rb') as fo:
        buf = fo.read()
        index=0
        if kind=='data':
        #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，需要读取前4行数据，所以需要4个i
            header = np.frombuffer(buf, '>i', 4, index)
            index += header.size * header.itemsize
            data = np.frombuffer(buf, '>B', header[1] * header[2] * header[3], index).reshape(header[1], -1)
        elif kind=='lable':
        #因为数据结构中前2行的数据类型都是32位整型，所以采用i格式，需要读取前2行数据，所以需要2个i
            header=np.frombuffer(buf,'>i',2,0)
            index+=header.size*header.itemsize
            data = np.frombuffer(buf, '>B', header[1], index)
    return data
#--------------------加载数据-------------------                
X_train = load_data('train-images-idx3-ubyte.gz','data')  # 训练数据集的样本特征
y_train = load_data('train-labels-idx1-ubyte.gz','lable')  # 训练数据集的标签
X_test = load_data('t10k-images-idx3-ubyte.gz','data')  # 测试数据集的样本特征
y_test = load_data('t10k-labels-idx1-ubyte.gz','lable')  # 测试数据集的标签
#---------查看数据的格式-------------
print('Train data shape:')
print(X_train.shape, y_train.shape)
print('Test data shape:')
print(X_test.shape, y_test.shape)
#--------------查看几个数据-----------
index_1 = 1024
plt.imshow(np.reshape(X_train[index_1], (28, 28)), cmap='gray')
plt.title('Index:{}'.format(index_1))
plt.show()
print('Label: {}'.format(y_train[index_1]))

index_2=2048
plt.imshow(np.reshape(X_train[index_2], (28, 28)), cmap='gray')
plt.title('Index:{}'.format(index_2))
plt.show()
print('Label: {}'.format(y_train[index_2]))
