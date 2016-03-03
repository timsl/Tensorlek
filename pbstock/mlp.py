from inputman import Inputman
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.connections import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
import numpy as np

training_range = 1000
testing_range = 100
training_start = 500
n_input = 2


ds = SupervisedDataSet(n_input, 1) 

min_price = 250
max_price = 1500

iman = Inputman(n_input, 1)
iman.set_interval(min_price, max_price)
iman.set_norm_range(-1,1)
iman.set(training_start)


for x in range(0, training_range): 
    xs, ys = iman.next_norm()
    ds.addSample(xs[0], ys[0])

nn = FeedForwardNetwork()
inLayer = LinearLayer(n_input)
hiddenLayer = TanhLayer(25)
outLayer = LinearLayer(1)

nn.addInputModule(inLayer)
nn.addModule(hiddenLayer)
nn.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
nn.addConnection(in_to_hidden)
nn.addConnection(hidden_to_out)
nn.sortModules()

trainer = BackpropTrainer(nn, ds, learningrate=0.01, momentum=0.1)

for epoch in range(0, 10000000):
    if epoch % 1000000 == 0:
        error = trainer.train()
        print('Epoch: ', epoch)
        print('Error: ', error)


result = []
real = []
# Testing
for i in range(0, testing_range): 
    xs, ys = iman.next_norm()
    result.append(nn.activate(xs[0]))
    real.append(ys[0])

plt.plot(result, 'r--', label='Predicted')
plt.plot(real, label='Real Data')
plt.legend(loc='best')
plt.show()

