import tensorflow as tf
from tensorflow.python import keras
from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint   
from keras.backend.tensorflow_backend import set_session

import os 
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import Utilities.Visuals as Visuals
from definitions import ROOT_DIR
from Utilities.MeasureDuration import MeasureDuration
from sklearn.metrics import confusion_matrix
from preprocessor import Loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
tf_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

SAVED_MODELS_FOLDER = os.path.join(ROOT_DIR, "Generated", "saved-models")

nodes = int(sys.argv[1])
epochs = int(sys.argv[2])

print ("Training model with {} nodes and for {} epochs..".format(nodes, epochs))

loaded_data = Loader.PreProcess()

train_X, train_Y = next(loaded_data.train_generator)
print (train_X.shape)
print (train_Y.shape)

base_mlp_model = Sequential()
base_mlp_model.add(Dense(nodes, input_shape=train_X.shape[1:], activation='relu'))
base_mlp_model.add(Dropout(0.2))
base_mlp_model.add(Dense(nodes*2, activation='relu'))
base_mlp_model.add(Dropout(0.3))
base_mlp_model.add(Dense(nodes*2, activation='relu'))
base_mlp_model.add(Dropout(0.3))
base_mlp_model.add(Flatten())
base_mlp_model.add(Dense(len(loaded_data.prediction_labels), activation='softmax'))
base_mlp_model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
base_mlp_model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

with MeasureDuration() as m:
    print ("Started training base MLP model...")
    best_model_file_path = os.path.join(SAVED_MODELS_FOLDER, "base-mlp-model.weights.nodes{}.epochs{}.best.hdf5".format(str(nodes), str(epochs)))
    checkpointer = ModelCheckpoint(
        best_model_file_path,
        verbose = 1,
        save_best_only = True)

    base_mlp_model.fit_generator(loaded_data.train_generator,
        steps_per_epoch = train_X.shape[0],
        epochs = epochs,
        validation_data = loaded_data.val_generator,
        callbacks = [checkpointer],
        verbose = 2)

    print ("Training base MLP model complete!")

# Load up the testing dataset
test_X, test_Y = next(loaded_data.test_generator)

# Store the Testing data for separate analysis, so that we do not test on trained data
np.savez(os.path.join(SAVED_MODELS_FOLDER, "base-mlp-model.testdata.nodes{}.epochs{}.npz".format(str(nodes), str(epochs))),
    test_X=test_X, 
    test_Y=test_Y, 
    prediction_labels=loaded_data.prediction_labels)
