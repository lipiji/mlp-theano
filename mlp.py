#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from softmax_layer import *
from logistic_layer import *
from updates import *

class MLP(object):
    def __init__(self, optimizer, in_size, out_size, hidden_size):
        X = T.matrix("X")
        self.create_model(optimizer, X, in_size, out_size, hidden_size)

    def create_model(self, optimizer, X, in_size, out_size, hidden_size):
        self.layers = []
        self.X = X
        self.n_hlayers = len(hidden_size)
        self.params = []
        self.optimizer = optimizer

        for i in xrange(self.n_hlayers):
            if i == 0:
                layer_input = X
                shape = (in_size, hidden_size[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (hidden_size[i - 1], hidden_size[i])

            hidden_layer = LogisticLayer(shape, layer_input)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        output_layer = SoftmaxLayer((hidden_layer.out_size, out_size), hidden_layer.activation)
        self.layers.append(output_layer)

        self.params += output_layer.params

        self.create_funs(X)
    
    def create_funs(self, X):
        activation = self.layers[len(self.layers) - 1].activation
        Y = T.matrix("Y")
        cost = T.nnet.categorical_crossentropy(activation, Y).mean()
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, None, None, lr)

        self.train = theano.function(inputs = [X, Y, lr], outputs = cost, updates = updates)
        self.predict = theano.function(inputs = [X], outputs = activation)
    
