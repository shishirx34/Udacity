import os
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import definitions
from Utilities.MeasureDuration import MeasureDuration
import Utilities.Visuals as Visuals

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nodes = sys.argv[1]
epochs = sys.argv[2]

best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "base-mlp-model.weights.nodes{}.epochs{}.best.hdf5".format(nodes, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "base-mlp-model.testdata.nodes{}.epochs{}.npz".format(nodes, epochs))

# Load weights with best validation score
print("Loading model {} and test data {}...".format(best_model_file_path, model_testdata_file_path))
base_mlp_model = load_model(best_model_file_path)
testdata = np.load(model_testdata_file_path)

prediction_labels = testdata['prediction_labels']
test_X = testdata['test_X']
test_Y = testdata['test_Y']

# Predict the diseases on testing set with the trained model
pred_Y = base_mlp_model.predict(test_X, batch_size=100, verbose=2)

# Check the mean of positive predictions by the model for each disease
print ("Positive predictions for diseases...")
for label, t_count, p_count in zip(prediction_labels, 100*np.mean(test_Y, 0), 100*np.mean(pred_Y, 0)):
    print('%s: Tx: %2.2f%%, Px: %2.8f%%' % (label, t_count, p_count))

# Generate the ROC Curve with Area Under the Curve metrics
Visuals.draw_roc_auc("Base MLP ROC AUC", prediction_labels, test_Y, pred_Y)
plt.savefig(os.path.join(definitions.OUTPUT_GENERATED_IMAGES_FOLDER, "base_mlp_roc_auc.jpg"))
