import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from data import prepare_data
from models import *
import pandas as pd

(x_train, y_train), (x_test, y_test) = prepare_data(one_d=False)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# from models import MLP
# model = MLP()
# model = MLP_2
model = MLP_narrow

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='roc', curve='ROC'),
        tf.keras.metrics.AUC(name='pr', curve='PR'),
]


def train():

    CLASS_WEIGHTS = {
        0: 0.125,
        1: 1
    }
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=metrics)
    h = model.fit(train_ds, epochs=15, validation_data=test_ds,
                  use_multiprocessing=True, workers=8,
                  class_weight=CLASS_WEIGHTS
                  )

    df = pd.DataFrame(h.history)
    print(df)


if __name__ == '__main__':
    train()
