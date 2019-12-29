import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
print("Imported all tools")
inp = open("data.txt", "r")

n = int(inp.readline())

inp_x = []
inp_y = []
for i in range(n):
    i_x, i_y = list(map(float, inp.readline().split()))
    inp_x.append(i_x)
    inp_y.append(i_y)

data_x = np.array(inp_x, dtype=float)
data_y = np.array(inp_y, dtype=float)

print("data include:")
for i in range(n):
    print("x = {} and y = {}".format(data_x[i], data_y[i]))

layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([
    layer_0
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(data_x, data_y, epochs=500, verbose=False)
print("Finished training the model")

print("These are the layer variables: {}".format(layer_0.get_weights()))

for i_x, i_y in [(55, 131), (17, 62.6), (93, 199.4)]:
    print("With x = {}, the model gives a result of {} while y = {}".format(i_x, model.predict([i_x])[0][0], i_y))