import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dataset_dir = "mnist"
batch_size = 16
lr = 0.1
input_dimension = 784

x = tf.placeholder(tf.float32, [None, input_dimension])
y = tf.placeholder(tf.int32, [None, 10])

z1 = tf.layers.dense(x, 392)
a1 = tf.nn.relu(z1)

z2 = tf.layers.dense(a1, 196)
a2 = tf.nn.relu(z2)

z3 = tf.layers.dense(a2, 98)
a3 = tf.nn.relu(z3)

z4 = tf.layers.dense(a3, 4)
a4 = tf.nn.relu(z4)

z5 = tf.layers.dense(a4, 3)
a5 = tf.nn.relu(z5)

z6 = tf.layers.dense(a5, 10)
h = tf.nn.relu(z6)

w = tf.trainable_variables()

loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, h))
grad = tf.gradients(loss, w)
op = tf.train.GradientDescentOptimizer(lr)
train = op.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

step = 1000

data_sets = input_data.read_data_sets(dataset_dir, one_hot=True)

for i in range(step):
    print("step %d : =====================================" % i)
    train_x, train_y = data_sets.train.next_batch(batch_size)

    _, z_, a_, z2_, w_, loss_, grad_ = sess.run([train, z1, a1, z2, w, loss, grad], feed_dict={x: train_x, y: train_y})
    print("loss", loss_)
    print(grad_[-4])
