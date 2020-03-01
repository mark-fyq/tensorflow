#shenjingtest4.py
   
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()#启用动态图机制

state = tf.Variable(0,name = 'counter')

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()  #所有变量初始化

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        
        