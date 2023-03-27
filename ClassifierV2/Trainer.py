import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import TensorBoard
def preprocess_image(image, label):
    # convert [0, 255] range integers to [0, 1] range floats
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

print ("Tensorflow version : " + tf.__version__)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname = 'flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Found " + str(image_count) + " images in " + str(data_dir))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

batch_size = 64
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2,subset="validation",seed=123,image_size=(img_height, img_width), batch_size=batch_size)
class_names = train_ds.class_names
print("train_ds before shuffle and repeat : " + str(len(train_ds)))
print("val_ds before shuffle and repeat : " + str(len(val_ds)))
train_ds = train_ds.repeat().shuffle(128)
val_ds = val_ds.repeat().shuffle(128)

#train_ds = train_ds.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
#train_ds = train_ds.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

logdir = os.path.join("logs", "Classifier_V2")
tensorboard = TensorBoard(log_dir=logdir)
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
  callbacks =[tensorboard],
  steps_per_epoch = 90,
  validation_steps = 90
)
model.save("ClassiferV2.h5")