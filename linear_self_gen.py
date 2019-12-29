import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
print("Imported all tools")

# y = Ax + B
A, B = list(map(float, input("Fill in two coefficients ").split()))

inp_x = []
inp_y = []

n = int(input("Number of data to train "))
for i in range(n):
    rng = random.random()
    i_x = rng*100
    i_y = i_x*A + B
    inp_x.append(i_x)
    inp_y.append(i_y)

data_x = np.array(inp_x, dtype=float)
data_y = np.array(inp_y, dtype=float)

show_gen = input("Show generated data ? (y/n)")

if show_gen == "y":
    print("Generated data include:")
    for i in range(n):
        print("x = {} and y = {}".format(data_x[i], data_y[i]))


layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([
    layer_0
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
epo = int(input("Fill number of iterations "))
print("Training...")
model.fit(data_x, data_y, epochs=epo, verbose=False)

print("Finished training the model")

print("Model has A = {}, B = {}".format(layer_0.get_weights()[0][0][0], layer_0.get_weights()[1][0]))
"""
while True:
    test_x = int(input("Fill in the test value "))
    if test_x == 67:
        break
    test_y = test_x*A + B
    print("Desired value: {}".format(test_y))
    print("Model prediction: {}".format(model.predict([test_x])[0][0]))
"""

export_path_keras = "./{}.h5".format("saved_model")
print(export_path_keras)

model.save(export_path_keras)
print("saved")
