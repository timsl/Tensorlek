
import numpy as np
from GAfuncs import *

"""
p1 = np.random.randint(10, size=(5,5))
p2 = np.random.randint(10, size=(5,5))

ga = GAfuncs(5,5)

print("Parent 1:\n")
print(p1)

print("\n\nParent 2:\n")
print(p2)

print("\n\nChild:\n")
child = ga.crossover(p1, p2)
print(child)
print(ga.mutate(child, 0.1))
"""


ga = GAfuncs(1,5)

ws = np.random.normal(1, 5, size=(5,1))
bs = np.random.normal(1, 5, size=(1,1))
xs = [[1., 2., 3., 4., 5.]]
ys = [[1., 2., 1.3, 3.75, 2.25]]


def pred():
    xs * ws + bs 

def fitfunc(Y, Y_):
    return np.mean(xs*ws + bs) 

old_fit = fitfunc(ys, pred)

for gen in range(10000000):
    m_ws = ga.mutate(ws,1)
    m_bs = bs * np.random.normal( 1, 1)

    fit = fitfunc(ys, pred)

    if gen % 500 == 0:
    #    print(m_ws)
        #print(m_bs)
        print(old_fit)
        #print(fit)
    if fit < old_fit:
        ws = m_ws
        bs = m_bs
        old_fit = fit
        print(">>>>>>>", gen, " ", fit)
