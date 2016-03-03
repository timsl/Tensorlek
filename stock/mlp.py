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
n_hidden_1 = 5
n_hidden_2 = 3

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
y_s = tf.scalar_summary("30_price", tf.reduce_sum(y))


with tf.name_scope('weights'):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, w_stddev)) ,
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, w_stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_1, 1], 0, w_stddev)),
    }
with tf.name_scope('biases'):
    biases = {
        #'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, b_stddev)),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, b_stddev)),
        #'out': tf.Variable(tf.random_normal([1], 0, b_stddev))
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_output]))
    }

w1_hist = tf.histogram_summary("w1", weights['h1'])
w2_hist = tf.histogram_summary("w2", weights['h2'])
wout_hist = tf.histogram_summary("wout", weights['out'])
b1_hist = tf.histogram_summary("b1", biases['b1'])
b2_hist = tf.histogram_summary("b2", biases['b2'])
bout_hist = tf.histogram_summary("bout", biases['out'])


with tf.name_scope('net'):
    # Create model
    def multilayer_perceptron(_X, _weights, _biases):
        with tf.name_scope('L1'):
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
        with tf.name_scope('L2'):
            layer_2 = tf.nn.sigmoid((tf.matmul(layer_1, _weights['h2'])+ _biases['b2']))
        with tf.name_scope('L3'):
            return tf.matmul(layer_1, _weights['out']) + _biases['out']/1e6

    pred = multilayer_perceptron(x, weights, biases)
    pred_s = tf.scalar_summary("20_pred", tf.reduce_sum(pred))

with tf.name_scope('net_tools'):
    with tf.name_scope('training'):
        #TODO: fixa "riktig" L2?
        cost = tf.reduce_sum(tf.pow(pred-y, 2))# L2
        cost_s = tf.scalar_summary("40_cost", cost)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Grad desc
        #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
    with tf.name_scope('evaluation'):
        accuracy = tf.truediv(tf.abs(y-pred),((y+pred)/2))*100
        accuracy_s = tf.scalar_summary("10_accuracy", tf.reduce_sum(accuracy))
      

merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()

for param in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    n_hidden_1 = param
    paramname = "LR"
    with tf.Session() as S:
        S.run(init)
        #writer = tf.train.SummaryWriter("./logs/" + str(hyperparams), S.graph_def)
        writer = tf.train.SummaryWriter("./logs/"+paramname+"/"+ str(param), S.graph_def)
        iman = Inputman(n_input,1)
        iman.set_interval(min_price, max_price)
        iman.set_norm_range(-1, 1)
        iman.set(training_start)
        avg_err = []
        for i in range(training_range):
            xs, ys = iman.next_norm()
            print(xs)
            if len(xs) == 0:
                print("No more data!")
                break
            res = S.run([merged, optimizer, cost, accuracy], feed_dict={x: xs, y: ys})
            avg_err.append(res[3])
            #print("Training step ", i, "\tcost: ", res[2])
            if (i % 50) == 0:
                print(np.mean(avg_err))
                avg_err = []
            writer.add_summary(res[0], i)
            
    
        print("Training done!")
        avg_err = []
        for i in range(testing_range):
            x_t, y_t = iman.next_norm()
            if len(x_t) == 0:
                print("No more data!")
                break 
            #print("----------------")
            res = S.run([merged, pred, accuracy], feed_dict={x: x_t, y: y_t})
            avg_err.append(res[2])
            #print("Training step ", i, "\tcost: ", res[2])
            if (i % 5) == 0:
                print(np.mean(avg_err))
                avg_err = []
            writer.add_summary(res[0], training_range+i)

