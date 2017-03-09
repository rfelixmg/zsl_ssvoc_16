
import theano
import theano.tensor as T
import os

os.environ['THEANO_FLAGS'] += ",exception_verbosity=high"

# define tensor variables
X = T.matrix("X")
Y = T.matrix("Y")
d = T.scalar('d')

from keras import backend as K

# euclidean_distance = lambda x, y: T.sqrt(T.sum(T.square(x - y), axis=2))
#
# results, updates = theano.scan(fn=euclidean_distance, outputs_info=[d])
# x = np.zeros((40, 300), dtype=theano.config.floatX)
# y = np.zeros((40,300), dtype=theano.config.floatX)
#
# print euclidean_distance(x, y)

#X = T.tensor3('X')


import theano
import theano.tensor as T
import numpy as np
from keras import backend as K

X = T.matrix('X')
W = T.matrix('W')


from utils.experiments_util import get_fake_semantic_embedding, collect_fake_splits

Ws, Wt = get_fake_semantic_embedding()
Ws = K.variable(Ws)
Wt = K.variable(Wt)

def d(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=1))


def batch_euclidean_distance(x, W):
    return K.sqrt(K.sum(K.square(x - W), axis=1))


def batch_processing(X):
    def _batch_itr(x):
        Ds = batch_euclidean_distance(x, Ws)
        Dt = batch_euclidean_distance(x, Wt)
        return Ds, Dt

    output, updates = theano.scan(fn=_batch_itr, sequences=[X])
    return output

#results, updates = theano.scan(fn=lambda x, W, b_sym: T.tanh(T.dot(x,W) + b_sym), sequences=X, non_sequences=[W, b_sym])
results, updates = theano.scan(fn=lambda x, w: batch_euclidean_distance(x, w), sequences=[X], non_sequences=[W])
euclidean_distance = theano.function(inputs=[X, W], outputs=[results])

#results, updates = theano.scan(fn=lambda x, w: self.batch_euclidean_distance(x, w), sequences=[sym_X],
#                               non_sequences=[sym_W])
#self.euclidean = theano.function(inputs=[sym_X, sym_W], outputs=[results])

X_train, A_train, Y_train, X_test, A_test, Y_test, knn_feat, labels_test = collect_fake_splits(200)

# x = np.zeros((32, 300), dtype=theano.config.floatX)
#
# result = batch_processing(A_train)

#x = np.array([x,x])
#w = np.ones((40,300), dtype=theano.config.floatX)
#b = np.ones((2), dtype=theano.config.floatX)

#result = euclidean_distance(x, w)[0]


