import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt  #调用绘图函数

tf.disable_eager_execution()   #启用动态图机制

def add_layer(inputs,in_size,out_size,activation_function=None):   #建立层（输入数据，输入数据的列数，输出数据的列数，激活函数）
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))    #生成一个in_size行，out_size列的随机变量矩阵
    #这个操作返回一个具有shape形状的dtype类型的张量，所有元素都设置为零。但是全返回0可能会报错故加上0.1 。
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)      
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

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1)

plt.pause(0)
