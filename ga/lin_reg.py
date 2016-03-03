#! /usr/bin/python

import tensorflow as tf
import numpy as np
from GAfuncs import *

X = tf.placeholder("float", [1, 5])
Y = tf.placeholder("float", [1, 5])


W = tf.placeholder("float", [5,1])
b = tf.placeholder("float", [1,5])


Y_p = tf.matmul(X, W) + b



fitness = tf.reduce_mean(tf.abs(Y-Y_p))










with tf.Session() as S:
    S.run(tf.initialize_all_variables())
    ws = np.random.normal(0, 0.01, size=(5,1))
    bs = np.random.normal(0, 0.01, size=(1,5))
    xs = [[1., 2., 3., 4., 5.]]
    ys = [[1., 2., 1.3, 3.75, 2.25]]
    ga = GAfuncs(5,5)

    old_fit = S.run(fitness, feed_dict={X: xs, Y: ys, W: ws, b: bs})
    for gen in range(10000):
        m_ws = ga.mutate(ws, 1)
        m_bs = ga.mutate(bs, 1)
        fit = S.run(fitness, feed_dict={X: xs, Y: ys, W: m_ws, b: m_bs})
        if gen % 100 == 0:
            #print(np.mean(m_ws))
            #print(gen, "\t", fit)
            print(Y_p.eval(feed_dict={X: xs, Y: ys, W: m_ws, b: m_bs}) )
        if fit < old_fit:
            ws = m_ws
            bs = m_bs
            old_fid = fit
            print("Gen: %i15\t fitness: %d" % (gen, fit))

