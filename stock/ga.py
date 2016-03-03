import tensorflow as tf
import numpy as np

from inputman import Inputman

learning_rate = 0.4
training_start = 500
training_range = 2000
testing_range = 300

min_price = 250
max_price = 1400 #NAJS PRAJS

n_input = 5
n_output = 1
n_hidden = 5

w_stddev = 0.01
b_stddev = 0.00001


hyperparams = {
    'LR'    :   learning_rate,
    'TrS'   :   training_start,
    'TrR'   :   training_range,
    'TeR'   :   testing_range,
    'n_i'   :   n_input,
    'n_o'   :   n_output,
    'n_h_1' :   n_hidden_1,
    'n_h_2' :   n_hidden_2,
    'w_s'   :   w_stddev,
    'b_s'   :   b_stddev
}

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


h1 = tf.Variable(tf.random_normal([n_input, n_hidden], 0, w_stddev))
b1 = tf.Variable(tf.zeros([n_hidden]))

pred = tf.matmul(x, h1) + b1

cost = tf.reduce_sum(tf.pow(pred-y, 2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Grad desc

accuracy = tf.truediv(tf.abs(y-pred),((y+pred)/2))*100

accuracy_s = tf.scalar_summary("10_accuracy", tf.reduce_sum(accuracy))
      

merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()


params = [100, 0.05, 250, n_hidden, 20]
curPop = np.random.choice(np.arange(-15, 15, step=0.01), size=(params[0], params[3]), replace=False)
nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))
fitVec = np.zeros((params[0], 2))


with tf.Session() as S:
    S.run(init)
    iman = Inputman(n_input,1)
    iman.set_interval(min_price, max_price)
    iman.set_norm_range(-1, 1)
    iman.set(training_start)
    avg_err = []

    for i in range(100):
        xs, ys = iman.next_norm()
        if len(xs) == 0:
            print("No more data!")
            break
        fitVec = np.array([np.array([x, np.sum

        #res = S.run([merged, optimizer, cost, accuracy], feed_dict={x: xs, y: ys})
        avg_err.append(res[3])
        if (i % 50) == 0:
            print(np.mean(avg_err))
            avg_err = []
        
"""
    print("Training done!")
    avg_err = []
    for i in range(15):
        x_t, y_t = iman.next_norm()
        if len(x_t) == 0:
            print("No more data!")
            break 
        #print("----------------")
        res = S.run([merged, pred, accuracy], feed_dict={x: x_t, y: y_t})
        avg_err.append(res[2])
        if (i % 5) == 0:
            print(np.mean(avg_err))
            avg_err = []
"""
