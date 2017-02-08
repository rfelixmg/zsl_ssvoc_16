def euclidean_distance(y_true, y_pred):
    from keras.optimizers import K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1))

def distance(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_true - y_pred), axis=1))

def nothing_to_lose(y_true, y_pred):
    return y_pred

def mode_euclidean_distance(inputs):
    (y_true, y_pred) = inputs
    from keras.optimizers import K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=1))


from keras.layers import Input, Dense, Merge, Lambda, merge, Flatten
from keras.models import Model

import numpy as np

# First model (MLP from CNN -> semantic_embedding space)
input_x = Input(shape=(4096,))
f_x = Dense(300,
            name='W_embedding',
            activation='linear')(input_x)
visual = Model(input=input_x, output=f_x)

# Second model (linear transformation from GloVe -> semantic_embedding space)
input_u = Input(shape=(300,))
g_u = Dense(300,
            name='V_embedding',
            init='identity',
            activation='linear')(input_u)
semantic = Model(input=input_u, output=g_u)

# Layer to calc distance between generated vector on semantic embedding space
# Joint learning
distance = Lambda(mode_euclidean_distance, output_shape=(1,))([g_u, f_x])

model = Model(input=[input_u, input_x], output=distance)
# Compile model using function "nothing_to_lose"
# I create this function, it only returns the result from the euclidean_distance
model.compile(optimizer='sgd',
              loss=nothing_to_lose)

X = np.random.random((100,4096))
U = np.random.random((100,300))
Y = np.ones((100))

model.fit([U, X], Y, nb_epoch=10)