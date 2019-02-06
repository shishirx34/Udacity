import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from keras import Sequential
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint   
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD

import definitions
import Utilities.Visuals as Visuals
from Utilities.MeasureDuration import MeasureDuration
from preprocessor import Loader

def TrainChestR(model, 
    training_layer, 
    train_gen,
    train_X,
    val_gen,
    attempt, 
    epochs)
    with MeasureDuration() as m:
        print ("Started training ChestR InceptionV3 CNN model - {} layer...").format(training_layer)
        best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-transfer-cnn-model.weights.layer{}.att{}.epochs{}.best.hdf5".format(training_layer, attempt, epochs))
        checkpointer = ModelCheckpoint(
            best_model_file_path,
            verbose = 1,
            save_best_only = True)

        model.fit_generator(train_gen,
            steps_per_epoch = train_X.shape[0],
            epochs = epochs,
            validation_data = val_gen,
            callbacks = [checkpointer],
            verbose = 2)

        print ("Training ChestR InceptionV3 CNN model complete!")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

epochs = int(sys.argv[1])

print ("Training model for {} epochs..".format(epochs))

loaded_data = Loader.PreProcess()

train_X, train_Y = next(loaded_data.train_generator)
print (train_X.shape)
print (train_Y.shape)
'''
Attempt 1: with 128,128 Image size and InceptionV3 ImageNet model transfer learning
'''
attempt = 1

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=train_X.shape[1:])

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(Dense(len(loaded_data.prediction_labels), activation='softmax'), activation='softmax')(x)

# this is the model we will train
chestr_transfer_cnn_model = Model(inputs=base_model.input, outputs=predictions)

chestr_transfer_cnn_model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
chestr_transfer_cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
TrainChestR(model = chestr_transfer_cnn_model, 
    training_layer = "top", 
    train_gen = loaded_data.train_generator,
    train_X = train_X,
    val_gen = loaded_data.validation_generator,
    attempt = attempt, 
    epochs = epochs)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in chestr_transfer_cnn_model.layers[:249]:
   layer.trainable = False
for layer in chestr_transfer_cnn_model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
chestr_transfer_cnn_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
TrainChestR(model = chestr_transfer_cnn_model, 
    training_layer = "bottom", 
    train_gen = loaded_data.train_generator,
    train_X = train_X,
    val_gen = loaded_data.validation_generator,
    attempt = attempt, 
    epochs = epochs)

# Load up the testing dataset
test_X, test_Y = next(loaded_data.test_generator)

# Store the Testing data for separate analysis, so that we do not test on trained data
np.savez(os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-transfer-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs)),
    test_X=test_X,
    test_Y=test_Y, 
    prediction_labels=loaded_data.prediction_labels)
