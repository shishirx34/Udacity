from preprocessor import Loader
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint   
import os 
from definitions import ROOT_DIR
from Utilities.MeasureDuration import MeasureDuration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAVED_MODELS_FOLDER = os.path.join(ROOT_DIR, "Generated", "saved-models")

loaded_data = Loader.PreProcess()

train_X, train_Y = next(loaded_data.train_generator)

print (train_X.shape)
print (train_Y.shape)

model = Sequential()
model.add(Dense(512, input_shape=train_X.shape[1:], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(len(loaded_data.prediction_labels), activation='softmax'))
model.summary()

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

with MeasureDuration() as m:
    print ("Started training base MLP model...")
    epochs = 5
    best_model_file_path = os.path.join(SAVED_MODELS_FOLDER, "base-mlp-model.weights.best.hdf5")
    checkpointer = ModelCheckpoint(
        best_model_file_path,
        verbose = 1,
        save_best_only = True)

    model.fit_generator(loaded_data.train_generator,
        steps_per_epoch = train_X.shape[0],
        epochs = epochs,
        validation_data = loaded_data.val_generator,
        callbacks = [checkpointer],
        verbose = 2)

    print ("Training base MLP model complete!")
