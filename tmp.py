import os

os.environ['THEANO_FLAGS'] += ",exception_verbosity='high'"

import keras.backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.models import Model
import numpy as np

def nothing_to_lose(y_true, y_pred):
    return y_pred

class SigalLoss(Layer):
    def __init__(self,  Ws, Wt, C, alpha,**kwargs):
        self.Ws     = K.variable(Ws, name='Ws')
        self.Wt     = K.variable(Wt, name='Ws')
        self.C      = K.variable(C, name= 'C')
        self.alpha  = K.variable(alpha, name= 'C')
        super(SigalLoss, self).__init__(**kwargs)

    def euclidean_distance(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred), axis=0))

    def vector_distance(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred)))

    def build(self, input_shape):
        super(SigalLoss, self).build(input_shape)

    def call(self, inputs, mask=None):
        X, A = inputs

        D = self.vector_distance(X, A)

        Xt = X.repeat(self.Wt.shape[0].eval(), axis=0)
        Dt = self.euclidean_distance(self.Wt, Xt)

        Xs = X.repeat(self.Ws.shape[0].eval(), axis=0)
        Ds = self.euclidean_distance(self.Ws, Xs)

        MS = 0.5 * K.sum(self.C + 0.5 * D - 0.5 * Ds)
        MV = 0.5 * K.sum(self.C + 0.5 * D - 0.5 * Dt)

        return (self.alpha * D) + ((1 - self.alpha)*(MS + MV))

    def get_output_shape_for(self, input_shape):
        return (1,1)


input_x = Input(shape=(4096,))
input_u = Input(shape=(300,))

Ws = np.random.random((40,300))
Ws[0] = np.ones(300)
Wt = np.zeros((10,300))

visual = Dense(300,
               name='W_embedding',
               activation='linear')(input_x)

sigal_layer = SigalLoss(Ws=Ws, Wt=Wt, C=0.01, alpha=0.01)
f_x = sigal_layer([visual, input_u])

model = Model(input=([input_x, input_u]), output=f_x)
model.compile(loss=nothing_to_lose, optimizer='sgd')

X = np.random.random((100, 4096))
A = np.random.normal(0.5, 1, (100,300))
Y = np.zeros(100)

model.fit([X,A], Y, nb_epoch=10, batch_size=1)
