#pylint: skip-file
#https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
import numpy as np
import theano
import theano.tensor as T

def SGD(params, gparams, learning_rate = 0.1):
    updates = []
    for p, g in zip(params, gparams):
        updates.append((p, p - learning_rate * g))
    return updates

def RMSprop(params, gparams, learning_rate = 0.01, rho = 0.9, epsilon = 1e-6):
    updates = [] 
    for p, g in zip(params, gparams):
        v = p.get_value(borrow=True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g / T.sqrt(acc_new + epsilon)))
    return updates
