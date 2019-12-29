from __future__ import absolute_import, division, print_function, unicode_literals
# Just disables the warning, doesn't enable AVX/FMA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Importing modules...")
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import math
import numpy as np
import time
import logging
import PIL
from PIL import Image
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print("Retrieving data")
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
test_dataset = dataset['test']

class_names = [_ for _ in range(10)]

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

test_dataset  =  test_dataset.map(normalize)
test_dataset  =  test_dataset.cache()

num_test_examples = metadata.splits['test'].num_examples
print("Number of test examples:     {}".format(num_test_examples))

model = tf.keras.models.load_model("mnist_1st.h5", custom_objects={'KerasLayer': hub.KerasLayer})

test_dataset = test_dataset.batch(100)

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/100))
print('Accuracy on test dataset:', test_accuracy)

def make_prediction(img):
    #set img size
    #img = img.numpy().reshape((28, 28))
    img = np.array([img])
    prediction = model.predict(img)
    answer = np.argmax(prediction)
    print(prediction)
    print("The model gives a prediction of {}".format(answer))

image = Image.open("3.png").resize((28, 28)).convert('L')
image = np.array(image)/255.0
image = np.expand_dims(image, axis=2)
print(type(image))
print(image.shape)

make_prediction(image)
