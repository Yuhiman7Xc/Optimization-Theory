# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:22:12 2019
@author: Ding
"""
import numpy as np
import matplotlib.pyplot as plt
import gzip as gz
import time
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



#-------------------------------定义常量-----------------
x_dim = 28 * 28  #data shape,28*28
y_dim = 10       #output shape,10*1
W_dim = (y_dim, x_dim)  # 矩阵W
b_dim = y_dim     #index b
#----------------------定义logic regression中的必要函数------
# Softmax函数
def softmax(x):
    """
    输入向量x,输出其经softmax变换后的向量
    """
    return np.exp(x) / np.exp(x).sum()

#Loss函数
#L=-log(Si(yi)),yi为预测值
def loss(W, b, x, y):
    """
    W, b为当前的权重和偏置参数,x,y为样本数据和该样本对应的标签
    """
    return -np.log(softmax(np.dot(W,x) + b)[y])  # 预测值与标签相同的概率

#单样本损失函数梯度
#按照公式很容易写出来
def L_Gra(W, b, x, y):
    """
     W, b为当前的权重和偏置参数,x,y为样本数据和该样本对应的标签
    """
    #初始化
    W_G = np.zeros(W.shape)
    b_G = np.zeros(b.shape)
    S = softmax(np.dot(W,x) + b)
    W_row=W.shape[0]
    W_column=W.shape[1]
    b_column=b.shape[0]
    #对Wij和bi分别求梯度
    for i in range(W_row):
        for j in range(W_column):
            W_G[i][j]=(S[i] - 1) * x[j] if y==i else S[i]*x[j]
               
    for i in range(b_column):
        b_G[i]=S[i]-1 if y==i else S[i]
    #返回W和b的梯度    
    return W_G, b_G

#检验模型在测试集上的正确率
def test_accurate(W, b, X_test, y_test):
    num = len(X_test)
    #results中保存测试结果，正确则添加一个1，错误则添加一个0
    results = []             
    for i in range(num):
        #检验softmax(W*x+b)中最大的元素的下标是否为y，若是则预测正确
        y_i = np.dot(W,X_test[i])+ b
        res=1 if softmax(y_i).argmax()==y_test[i] else 0
        results.append(res)
        
    accurate_rate=np.mean(results)
    return accurate_rate



# mini-batch随机梯度下降
def mini_batch(batch_size,alpha,epoches):
    """ 
    batch_size为batch的尺寸,alpha为步长,epochsnum为epoch的数量,即直接决定了训练的次数
    """
    accurate_rates = []  # 记录每次迭代的正确率 accurate_rate
    iters_W = []  # 记录每次迭代的 W
    iters_b = []  # 记录每次迭代的 b
    
    W = np.zeros(W_dim)  
    b = np.zeros(b_dim) 
    #根据batch_size的尺寸将原样本和标签分批后
    #初始化
    x_batches=np.zeros(((int(X_train.shape[0]/batch_size),batch_size, 784)))
    y_batches=np.zeros(((int(X_train.shape[0]/batch_size),batch_size)))
    batches_num = int(X_train.shape[0]/batch_size)
    #分批
    for i in range(0,X_train.shape[0],batch_size):
        x_batches[int(i/batch_size)]=X_train[i:i+batch_size]
        y_batches[int(i/batch_size)]=y_train[i:i+batch_size]
    print('Start training...')

    start = time.time()  # 开始计时
	#print(start)
    for epoch in range(epoches): #对所有样本循环一遍为一个epoch
        print("epoch: ")
        print(epoch)
        for i in range(batches_num): #对一个batch循环一遍为一个iteration
            # print("i: ")
            # print(i)
            #初始化梯度
            W_gradients = np.zeros(W_dim)
            b_gradients = np.zeros(b_dim)
            
            x_batch,y_batch= x_batches[i],y_batches[i]
            
            #求一个batch的梯度，实际上对一个batch中的样本梯度求和然后平均即可
            for j in range(batch_size):
                W_g,b_g = L_Gra(W, b,x_batch[j], y_batch[j])
                W_gradients += W_g
                b_gradients += b_g
            W_gradients /= batch_size  
            b_gradients /= batch_size
            #进行梯度下降
            W -= alpha * W_gradients
            b -= alpha * b_gradients
            #把迭代后的精度、W、b都加入到相应的数组，便于之后分析和绘制图像
            #当size较小时采用每100个ite记录一次
            #当size较大时每次ite记录一次
            # if i%100==0:
            #     accurate_rates.append(test_accurate(W, b, X_test, y_test))
            accurate_rates.append(test_accurate(W, b, X_test, y_test))
            iters_W.append(W.copy())
            iters_b.append(b.copy())
            
    end = time.time()  # 结束计时
    #print(end)
    time_cost=(end - start)
    
    return W,b,time_cost, accurate_rates, iters_W, iters_b


#-------------模型的主函数-------------------
def run(alpha,batch_size,epochs_num):
    #将参数带入并训练，将训练结果存储到下列变量中
    #W和b表示最优的W和b,time_cost表示训练该模型时间
    #accuracys,W_s,b_s分别表示accuracy,W,b随着迭代次数的变化的变化
    W,b,time_cost,accuracys,W_s,b_s = mini_batch(batch_size,alpha,epochs_num)
    
    #求W和b与最优解的距离，用二阶范数表示距离
    iterations=len(W_s)
    dis_W=[]
    dis_b=[]
    
    #test=b_s[:10]
    #print(test)
    #起初b的变化是直线，仔细分析数据后发现b值确实为非常小的值，图像中难以显示
    
    for i in range(iterations):   
        dis_W.append(np.linalg.norm(W_s[i] - W))
        dis_b.append(np.linalg.norm(b_s[i] - b))

    #--------简单介绍------
    print("the parameters is: step length alpah:{}; batch size:{}; Epoches:{}".format(alpha, batch_size, epochs_num))
    print("Result: accuracy:{:.2%},time cost:{:.2f}".format(accuracys[-1],time_cost))
        
    #----------------作图--------------
    #精确度随迭代次数的变化
    plt.title('The Model accuracy variation chart ')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(accuracys,'m')
    plt.grid()
    plt.show()
    #W和b距最优解的距离随迭代次数的变化
    plt.title('The distance from the optimal solution')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.plot(dis_W,'r', label='distance between W and W*')
    plt.plot(dis_b,'g',label='distance between b and b*')
    plt.legend()
    plt.grid()
    plt.show()
    
    #print(W,b)
    
#----------参数输入---------------
alpha = 1e-5
batch_size = 1000
epochs_num = 2
# 运行函数
run(alpha,batch_size,epochs_num)
