#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda

# set use gpu programatically
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_gradws(shape):
    return theano.shared(floatX(np.zeros(shape)))

def init_bias(size):
    return theano.shared(floatX(np.zeros((size,))))

def rmse(py, y):
    e = 0
    for t in xrange(len(y)):
        e += np.sqrt(np.mean((np.asarray(py[t,]) - np.asarray(y[t,])) ** 2))
    return e / len(y)

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
