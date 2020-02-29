#shenjingtest2.py
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()# 启用动态图机制
#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.3 + 0.8

###create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))#随机生成-1~1的数作为权重
biases = tf.Variable(tf.zeros([1]))#偏置初值为0

y = Weights*x_data + biases#预测的结果

loss = tf.reduce_mean(tf.square(y-y_data))#预测结果与实际结果的差值
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)#设置优化器和学习效率，数值越小每次改变的数值越小，最后的结果越精确
train = optimizer.minimize(loss)#减少误差

init = tf.compat.v1.global_variables_initializer()#变量初始化
###create tensorflow structure end ###

sess = tf.compat.v1.Session()#激活函数
sess.run(init)#激活变量初始化

for step in range(201):
	sess.run(train)#计算开始
	if step % 20 == 0:
		print(step,sess.run(Weights),sess.run(biases))#每计算20次打印一次权重和偏置