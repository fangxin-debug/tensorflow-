# tensorflow-
#导入数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"""
定义算法公式
"""
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros[10])

y = tf.nn.softmax(tf.matmul(x, W) + b)
"""
定义loss function，选择优化器，优化参数
"""
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reuce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))#????
#用随机梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
"""
迭代的训练数据
"""
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x:batch_xs, y:batch_ys})
"""
评测模型
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
