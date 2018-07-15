import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/Code/data_set/MNIST_data", one_hot=True)

imgrow = 28
imgcolunm = 28
learningrate = 1e-4
BATCH_SIZE = 1000

#def addConvLayer(input, filter, kernel, stride, padding, activation):
    

x = tf.placeholder(tf.float32, [None, imgrow * imgcolunm])
image = tf.reshape(x, [-1, imgrow, imgcolunm, 1])
y = tf.placeholder(tf.int32, [None, 10])

test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

conv1 = tf.layers.conv2d(
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
    name='firstlayer'
)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=2,
    strides=2,
)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
    name='secondlayer'
)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=2,
    strides=2,
)

flat = tf.reshape(pool2, [-1, 7*7*32])
midlayer = tf.layers.dense(flat, 300, activation=tf.nn.relu)
output = tf.layers.dense(midlayer,10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op) 

for step in range(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {x: b_x, y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {x: test_x, y: test_y})
        print('Step: ', step, ' loss: %.4f' % loss_, ' \naccuracy : %.4f' % accuracy_)
