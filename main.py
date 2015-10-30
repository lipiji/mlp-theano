#pylint: skip-file
import time
import numpy as np
import theano
import theano.tensor as T
from softmax_layer import *
from logistic_layer import *
import utils_pg as Utils
import data

#seqs, i2w, w2i = data.char_sequence()

lr = 0.2
batch_size = 200
datasets = data.shared_mnist()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

hidden_size = [400]
layers = []

#layers = [GRULayer(len(w2i), hidden_size[0]),
#          GRULayer(hidden_size[0], hidden_size[1]),
#          SoftmaxLayer(hidden_size[len(hidden_size) - 1], len(w2i))]

dim_x = train_set_x.get_value(borrow=True).shape[1]
dim_y = train_set_y.get_value(borrow=True).shape[1]
print dim_x, dim_y

np_train_x = np.asmatrix(train_set_x.get_value(borrow=True))
np_train_y = np.asmatrix(train_set_y.get_value(borrow=True))
np_valid_x = np.asmatrix(valid_set_x.get_value(borrow=True))
np_valid_y = np.asmatrix(valid_set_y.get_value(borrow=True))
np_test_x = np.asmatrix(test_set_x.get_value(borrow=True))
np_test_y = np.asmatrix(test_set_y.get_value(borrow=True))


for lay in xrange(len(hidden_size)):
    if lay == 0:
        shape = (dim_x, hidden_size[lay])
    else:
        shape = (hidden_size[lay - 1], hidden_size[lay])
    layers.append(LogisticLayer(shape))
layers.append(SoftmaxLayer((hidden_size[len(hidden_size) - 1], dim_y)))

nn = NN(layers)

#batch train
start = time.time()
for i in xrange(5):
    acc = 0.0
    in_start = time.time()
    for index in xrange(n_train_batches):
        X = np_train_x[index * batch_size: (index + 1) * batch_size, :]
        Y = np_train_y[index * batch_size: (index + 1) * batch_size, :]
        nn.batch_train(X, Y, lr)
    in_time = time.time() - in_start

    num_x = 0.0
    for index in xrange(n_valid_batches):
        X = np_valid_x[index * batch_size: (index + 1) * batch_size, :]
        Y = np_valid_y[index * batch_size: (index + 1) * batch_size, :]
        label = np.argmax(Y, axis=1)
        p_label = np.argmax(nn.predict(X), axis=1)
        for c in xrange(len(label)):
            num_x += 1
            if label[c] == p_label[c]:
                acc += 1

    print i, acc / num_x, in_time
print time.time() - start
