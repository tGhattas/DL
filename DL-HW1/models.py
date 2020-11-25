from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import hyper_params


class MLP(Model):

  def __init__(self, input_shape=(hyper_params.INPUT_DIM,), output_shape=hyper_params.OUTPUT_DIM):
    super(MLP, self).__init__()

    # self.flatten = Flatten()
    self.d1 = Dense(350, input_shape=input_shape, activation='relu')
    self.d2 = Dense(50, activation='relu')
    self.d3 = Dense(output_shape, activation='softmax')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return x
