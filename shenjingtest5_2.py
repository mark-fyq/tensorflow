import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt  #调用绘图函数

tf.disable_eager_execution()   #启用动态图机制

# def add_layer(inputs,in_size,out_size,activation_function=None):   #建立层（输入数据，输入数据的列数，输出数据的列数，激活函数）
    # Weights = tf.Variable(tf.random_normal([in_size,out_size]))    #生成一个in_size行，out_size列的随机变量矩阵这个操作返回一个具有shape形状的dtype类型的张量，所有元素都设置为零。但是全返回0可能会报错故加上0.1 。
    # biases = tf.Variable(tf.zeros([1,out_size])+0.1)      
    # Wx_plus_b = tf.matmul(inputs,Weights) + biases     #建立神经网络线性公式：inputs * Weights + biases
    # if activation_function is None:
        # outputs = Wx_plus_b    #如果没有设置激活函数，则直接就把当前信号原封不动地传递出去
    # else:
        # outputs = activation_function(Wx_plus_b)    #如果设置了激活函数，则会由此激活函数来对信号进行传递或抑制
    # return outputs
# 创建数据    
x_data = np.linspace(-1,1,300)[:,np.newaxis]    #生成一个-1~1数量为300的等差数列，并提升一个维度
noise = np.random.normal(0,0.05,x_data.shape)    #生成一个正太分布的噪声  
y_data = np.square(x_data) + noise     #生成一堆点

# 构建传入常量
xs = tf.placeholder(tf.float32,[None,1])    
ys = tf.placeholder(tf.float32,[None,1])

# l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# prediction = add_layer(l1,10,1,activation_function=None)
#构建中间层
Weights_l1 = tf.Variable(tf.random_normal([1,10]))    #因为输入1个数据，中间层有10个神经元所以输出为10，故为[1,10]
biases_l1 = tf.Variable(tf.zeros([1,10])+0.1)    #输入是1*input,Weights是input*output,output是1*output,所以biases是1*output
Wx_plus_b_l1 = tf.matmul(xs,Weights_l1) + biases_l1     #建立神经网络线性公式：inputs * Weights + biases
l1 = tf.nn.relu(Wx_plus_b_l1)    #中间层激活函数

#构建输出层
Weights_l2 = tf.Variable(tf.random_normal([10,1]))    #因为有10个神经元作为输入，而输出只有1个数据，故为[10,1]
biases_l2 = tf.Variable(tf.zeros([1,1])+0.1)    #输入是1*input,Weights是input*output,output是1*output,所以biases是1*output
Wx_plus_b_l2 = tf.matmul(l1,Weights_l2) + biases_l2     #建立神经网络线性公式：inputs * Weights + biases
prediction = Wx_plus_b_l2    #输出层是否使用激活函数看你选取的是哪个激活函数，如果是tf.nn.tanh则输出也需要激活函数tf.nn.tanh

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))    #计算一个张量的各个维度的元素之和的平均值。
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)    #有绘制多个图会用到，只生成一个图可以不写
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.5)
plt.pause(0)
