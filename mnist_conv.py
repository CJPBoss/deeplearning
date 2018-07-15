import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/Code/data_set/MNIST_data", one_hot=True)

imgrow = 28
imgcolunm = 28

def addConvLayer(input, filter, kernel, stride, padding, activation):
    

x = tf.placeholder(tf.float32, [None, imgrow * imgcolunm])
image = tf.reshape(x, [-1, imgrow, imgcolunm, 1])
y = tf.placeholder(tf.int32, [None, 10])

conv1 = tf.nn.conv2d(
    input=image,
    filter=5,
    strides=(1, 1),
    padding='same',
    data_format='NHWC',
    name='First Conv Layer'
)

