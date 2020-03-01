#shenjingtest3.py
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()#启用动态图机制

#构建两个常数

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)#矩阵相乘

##part1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()#关闭会话

#part2
with tf.Session() as sess:  #进入Session会话
    result = sess.run(product)
    print(result)

