import tensorflow as tf

# def create_mlp(num_layers=2, num_hidden=64):
#     model = tf.keras.Sequential()
#
#     # Add input layer
#     model.add(tf.keras.layers.InputLayer(input_shape=(19,)))
#
#     # Add hidden layers
#     for _ in range(num_layers):
#         model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))
#
#     # Add output layer
#     model.add(tf.keras.layers.Dense(3, activation=None))
#
#     return model


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(19,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation=None)
])
