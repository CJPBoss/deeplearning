import tensorflow as tf

a = tf.constant(12)
b = tf.constant(23)
c = tf.multiply(a, b)

sess = tf.Session()

print("==================================")
print(sess.run(c))