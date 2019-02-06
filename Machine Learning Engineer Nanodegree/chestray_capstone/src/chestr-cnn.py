import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from keras import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint   
from keras.backend.tensorflow_backend import set_session

import definitions
import Utilities.Visuals as Visuals
from Utilities.MeasureDuration import MeasureDuration
from preprocessor import Loader

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
Attempt 1: with 128,128 Image size and filters = 16
attempt = 1
chestr_cnn_model = Sequential()
chestr_cnn_model.add(Conv2D(filters=filters, kernel_size=2, padding='same', activation='relu', input_shape=train_X.shape[1:]))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Conv2D(filters=filters*2, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Conv2D(filters=filters*4, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Dropout(0.3))
chestr_cnn_model.add(Flatten())
chestr_cnn_model.add(Dense(512, activation='relu'))
chestr_cnn_model.add(Dropout(0.4))
chestr_cnn_model.add(Dense(len(loaded_data.prediction_labels), activation='softmax'))
'''
'''
Attempt 2: with 128,128 Image size
'''
attempt = 2
chestr_cnn_model = Sequential()
chestr_cnn_model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=train_X.shape[1:]))
chestr_cnn_model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Dropout(0.3))
chestr_cnn_model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(MaxPooling2D(pool_size=2))
chestr_cnn_model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
chestr_cnn_model.add(GlobalAveragePooling2D())
chestr_cnn_model.add(Dense(512, activation='relu'))
chestr_cnn_model.add(Dropout(0.4))
chestr_cnn_model.add(Dense(len(loaded_data.prediction_labels), activation='softmax'))

chestr_cnn_model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
chestr_cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

with MeasureDuration() as m:
    print ("Started training ChestR CNN model...")
    best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.weights.att{}.epochs{}.best.hdf5".format(attempt, epochs))
    checkpointer = ModelCheckpoint(
        best_model_file_path,
        verbose = 1,
        save_best_only = True)

    chestr_cnn_model.fit_generator(loaded_data.train_generator,
        steps_per_epoch = train_X.shape[0],
        epochs = epochs,
        validation_data = loaded_data.val_generator,
        callbacks = [checkpointer],
        verbose = 2)

    print ("Training ChestR CNN model complete!")

# Load up the testing dataset
test_X, test_Y = next(loaded_data.test_generator)

# Store the Testing data for separate analysis, so that we do not test on trained data
np.savez(os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs)),
    test_X=test_X,
    test_Y=test_Y, 
    prediction_labels=loaded_data.prediction_labels)
