import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt  #调用绘图函数

tf.disable_eager_execution()   #启用动态图机制

def add_layer(inputs,in_size,out_size,activation_function=None):   #建立层（输入数据，输入数据的列数，输出数据的列数，激活函数）
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))    #生成一个in_size行，out_size列的随机变量矩阵
    
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)    #这个操作返回一个具有shape形状的dtype类型的张量，所有元素都设置为零。但是全返回0可能会报错故加上0.1 。  
    #输入是1*input,Weights是input*output,output是1*output,所以biases是1*output
    Wx_plus_b = tf.matmul(inputs,Weights) + biases     #建立神经网络线性公式：inputs * Weights + biases
    if activation_function is None:
        outputs = Wx_plus_b    #如果没有设置激活函数，则直接就把当前信号原封不动地传递出去
    else:
        outputs = activation_function(Wx_plus_b)    #如果设置了激活函数，则会由此激活函数来对信号进行传递或抑制
    return outputs
#创建数据    
x_data = np.linspace(-1,1,300)[:,np.newaxis]    #生成一个-1~1数量为300的等差数列，并提升一个维度
noise = np.random.normal(0,0.05,x_data.shape)    #生成一个正太分布的噪声  
y_data = np.square(x_data) + noise     #生成一堆点

#构建传入常量
xs = tf.placeholder(tf.float32,[None,1])    
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)    #激活函数传入参数
prediction = add_layer(l1,10,1,activation_function=None)    #输出结果

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))    #计算一个张量的各个维度的元素之和的平均值。
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()    #初始化变量

fig = plt.figure()    #生成外框
ax = fig.add_subplot(1,1,1)    #生成1*1个图，选取第一个图
ax.scatter(x_data,y_data)    #绘制散点图
plt.ion()    #连续显示变化
plt.show()    #显示图像
with tf.Session() as sess:    #构建会话
    sess.run(init)    #激活初始化变量
    for i in range(2000):    #训练2000次
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})    
        if i % 50 == 0:    #没50次更新一次图
            try:    #抛出报错，即一旦出现exception类报错即执行except下的语句
                ax.lines.remove(lines[0])    #删除上一条线，因为最开始没有线所以会报错，采用try跳过错误
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})    #获取输出值
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)    #以x_data为x轴坐标，prediction_value为y轴坐标，画一条线宽为5，红色的实线
            plt.pause(0.1)    #画完暂停0.1秒

plt.pause(0)    #全部画完暂停，若没有这个语句显示完变化之后会直接关闭图框
