import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers
import keras_tuner as kt

def define_mlp(gas_shape_, num_of_classes_):
    initializer = tf.keras.initializers.HeNormal(seed=0)

    num_of_classes = num_of_classes_
    num_of_sensors = 16
    gas_shape = gas_shape_

    input_gas = keras.Input(shape=gas_shape)

    i = input_gas
    l = layers.Flatten()(i)
    l = layers.Dense(128, activation='tanh')(l)
    out = layers.Dense(num_of_classes)(l)

    model = tf.keras.Model(inputs=input_gas, outputs=out)

    model.summary()

    return model

