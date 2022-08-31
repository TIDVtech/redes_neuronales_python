# Import modules
from tabnanny import verbose
import tensorflow        as tf
import numpy             as np

# Create lists with random values
celsius    = np.array([-40, -10, 0, 8, 15, 22, 38],    dtype=float)
fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100], dtype=float)

# Neural Network
layer_input   = tf.keras.layers.Dense(units=32, input_shape=[1])
layer_hidden1 = tf.keras.layers.Dense(units=16)
layer_hidden2 = tf.keras.layers.Dense(units=4)
layer_output  = tf.keras.layers.Dense(units=1)

# Model
model = tf.keras.Sequential([layer_input, layer_hidden1, layer_hidden2, layer_output])

# Compile the model
model.compile(
  optimizer = tf.keras.optimizers.Adam(0.1),
  loss      = 'mean_squared_error'
)

# Train the model
history = model.fit(celsius, fahrenheit, epochs=60, verbose=False)

# Predict the value
print('''Hello and Welcome!
I can predict what's the value of Celsius in Fahrenheit :D''')
value  = float(input("Give me a value..."))
result = model.predict([value])
print(f'result is {result} Fahrenheit! above :P')