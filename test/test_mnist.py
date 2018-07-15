from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/Code/data_set/MNIST_data", one_hot=True)

a = mnist.train.images[0]
b = mnist.train.labels[0]
print("\n\n============================")
print(a)
print(b)