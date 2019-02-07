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
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Input, Dense, Dropout, Lambda, Flatten, Activation, BatchNormalization, regularizers, AvgPool2D, multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam

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
    epochs):
    with MeasureDuration() as m:
        print ("Started training ChestR VGG19 CNN model - {} layer...".format(training_layer))
        best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-vgg19-cnn-model.weights.layer{}.att{}.epochs{}.best.hdf5".format(training_layer, attempt, epochs))
        checkpointer = ModelCheckpoint(
            best_model_file_path,
            verbose = 1,
            save_best_only = True)

        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
        early = EarlyStopping(monitor="val_loss", 
                            mode="min", 
                            patience=10) # probably needs to be more patient, but kaggle time is limited
        callbacks_list = [checkpointer, early, reduceLROnPlat]

        model.fit_generator(train_gen,
            steps_per_epoch = train_X.shape[0],
            epochs = epochs,
            validation_data = val_gen,
            callbacks = callbacks_list,
            verbose = 2)

        print ("Training ChestR VGG19 CNN model complete!")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

epochs = int(sys.argv[1])

print ("Training model for {} epochs..".format(epochs))

# InceptionV3/VGG19 needs image to have 3 channels i.e. RGB channel, since our images are in 
# grayscale this won't work but our ImageDataGenerator can load the images in 'rgb' that could help
loaded_data = Loader.PreProcess(mode='rgb')

train_X, train_Y = next(loaded_data.train_generator)
print (train_X.shape)
print (train_Y.shape)

attempt = 3

# create the base pre-trained model
# Attempt 3 with imagenet weights
base_model = VGG19(weights='imagenet', include_top=False, input_shape=train_X.shape[1:])

# Attempt 4 with none weights
# base_model = VGG19(weights=None, include_top=False, input_shape=train_X.shape[1:])

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)

predictions = Dense(len(loaded_data.prediction_labels), activation='sigmoid')(x)

# lets not freeze the layers and see what happens since we have a huge number of images to train data on...
chestr_transfer_cnn_model = Model(inputs=base_model.input, outputs=predictions)

# Attempt 3 with SGD
chestr_transfer_cnn_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# Attempt 4 with ADAM optimizer
#chestr_transfer_cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy', 'mae'])
chestr_transfer_cnn_model.summary()

# train the model on the new data for a few epochs
TrainChestR(model = chestr_transfer_cnn_model, 
    training_layer = "all", 
    train_gen = loaded_data.train_generator,
    train_X = train_X,
    val_gen = loaded_data.val_generator,
    attempt = attempt, 
    epochs = epochs)

# Load up the testing dataset
test_X, test_Y = next(loaded_data.test_generator)

# Store the Testing data for separate analysis, so that we do not test on trained data
np.savez(os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-vgg19-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs)),
    test_X=test_X,
    test_Y=test_Y, 
    prediction_labels=loaded_data.prediction_labels)
