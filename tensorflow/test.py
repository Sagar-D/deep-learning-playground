import tensorflow as tf

x = tf.constant(2.0)
y = tf.constant(3.0)

print (x+y)
print(x*y)

x = tf.constant([[1,2],[3,4]], dtype=tf.float32)
y = tf.constant([[0,0],[1,1]], dtype=tf.float32)

print(x*y)
print(tf.matmul(x,y))