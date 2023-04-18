from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pathlib
import PIL
# hyper-parameters

batch_size = 64
img_height = 180
img_width = 180
# 10 categories of images (CIFAR-10)
num_classes = 5
# number of training epochs
epochs = 30

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname = 'flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Found " + str(image_count) + " images in " + str(data_dir))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

def load_data():
    """
    This function loads CIFAR-10 dataset, and preprocess it
    """
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    # loading the CIFAR-10 dataset, splitted between train and test sets
    ds_train = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    ds_test = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2,subset="validation",seed=123,image_size=(img_height, img_width), batch_size=batch_size)
    class_names = ds_train.class_names
    # repeat dataset forever, shuffle, preprocess, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test

def create_model(input_shape):
    # building the model
    model = Sequential()
    model.add(tf.keras.layers.Reshape(target_shape=(180, 180, 3)))
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape = (180, 180, 3)))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layer
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # print the summary of the model architecture
    
    # training the model using adam optimizer
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.summary()
    return model
if __name__ == "__main__":
    # load the data
    ds_train, ds_test = load_data()
    # constructs the model
    model = create_model(input_shape=3670)
    # some nice callbacks
    logdir = os.path.join("logs", "Flowers_V2")
    tensorboard = TensorBoard(log_dir=logdir)
    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    # train
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
              steps_per_epoch=30,
              validation_steps=30,
              callbacks=[tensorboard])
    # save the model to disk
    model.save("results/Flowers_V2.h5")