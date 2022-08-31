# Import modules
import tensorflow          as tf
import numpy               as np
import tensorflow_datasets as tfds
import matplotlib.pyplot   as plt
import math

# Download Dataset
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# Variables
train_data, test_data = data['train'], data['test']
class_names = metadata.features['label'].names

# Normalize and cache images
def normalize(image,label):
  image = tf.cast(image, tf.float32)
  image/= 255
  return image, label

train_data = train_data.map(normalize)
test_data  = test_data.map(normalize)

train_data = train_data.cache()
test_data  = test_data.cache()

# Neural Network
layer_input   = tf.keras.layers.Flatten(input_shape=(28,28,1))
layer_hidden1 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
layer_hidden2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
layer_output  = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

# Model
model = tf.keras.Sequential([layer_input, layer_hidden1, layer_hidden2, layer_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Preparing data to train the model
train_number = metadata.splits['train'].num_examples
test_number  = metadata.splits['test'].num_examples

LOOT_HEIGHT  = 32

train_data   = train_data.repeat().shuffle(train_number).batch(LOOT_HEIGHT)
test_data    = test_data.batch(LOOT_HEIGHT)

# Train the model
history = model.fit(train_data, epochs=50, steps_per_epoch=math.ceil(train_number/LOOT_HEIGHT))

# Print the Loss
plt.xlabel("# Epoch")
plt.ylabel("Loss maginutde")
plt.plot(history.history['loss'])

# Ready for predictions
for test_image, test_label in test_data.take(1):
  test_image = test_image.numpy()
  test_label = test_label.numpy()
  prediction = model.predict(test_image)

def graph_image(i, prediction_arr, real_label, image):
  prediction_arr, real_label, img = prediction_arr[i], real_label[i], image[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)
  prediction_label = np.argmax(prediction_arr)

  if prediction_label == real_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(
      class_names[prediction_label],
      100*np.max(prediction_arr),
      class_names[real_label],
      color=color
  ))

def graph_arr_value(i, predictions_arr, real_label):
  predictions_arr, real_label = predictions_arr[i], real_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  graphic = plt.bar(range(10), predictions_arr, color='#777777')
  plt.ylim([0,1])
  prediction_label = np.argmax(predictions_arr)

  graphic[prediction_label].set_color('red')
  graphic[real_label].set_color('blue')

files     = 5
columns   = 5
image_num = files * columns
plt.figure(figsize=(2*2*columns, 2*files))

for i in range(image_num):
  plt.subplot(files, 2*columns, 2*i+1)
  graph_image(i, prediction, test_label, test_image)
  plt.subplot(files, 2*columns, 2*i+2)
  graph_arr_value(i, prediction, test_label)

# Making manual predictions
value      = int(input("type the id of the image"))
image      = test_image[value]
image      = np.array([image])
prediction = model.predict(image)
print(f'Prediction is {class_names[np.argmax(prediction[0])]}')

# Save the model
model.save('prediction_model.h5')

# How to export for Tensorflowjs
# !pip install tensorflowjs
# !mkdir tfjs_target_dir
# !tensorflowjs_converter --input_format keras prediction_model.h5 tfjs_target_dir