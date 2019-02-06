import os
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import definitions
from Utilities.MeasureDuration import MeasureDuration
import Utilities.Visuals as Visuals

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
nodes = sys.argv[1]
epochs = sys.argv[2]
best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "base-mlp-model.weights.nodes{}.epochs{}.best.hdf5".format(nodes, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "base-mlp-model.testdata.nodes{}.epochs{}.npz".format(nodes, epochs))


epochs = sys.argv[1]
filters = sys.argv[2]
best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.weights.filters{}.epochs{}.best.hdf5".format(filters, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.testdata.filters{}.epochs{}.npz".format(filters, epochs))
'''

epochs = sys.argv[1]
attempt = 2
best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.weights.att{}.epochs{}.best.hdf5".format(attempt, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs))

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
    print('%s: Tx: %2.2f%%, Px: %2.2f%%' % (label, t_count, p_count))

print ("Accuracy of predictions...")
round_pred = pred_Y > 0.5
#print("Label \t Accuracy \t Precision \t Recall \t F1Score")
print("Label \t\t Accuracy")
for index, label in enumerate(prediction_labels):
    cm = confusion_matrix(test_Y[:,index], round_pred[:, index])
    print (cm)

#        100*precision_score(test_Y[:,index], round_pred[:, index]),
#        100*recall_score(test_Y[:,index], round_pred[:, index]),
#        f1_score(test_Y[:,index], round_pred[:, index])))


# Generate the ROC Curve with Area Under the Curve metrics
Visuals.draw_roc_auc("ChestR CNN ROC AUC", prediction_labels, test_Y, pred_Y)
plt.show()
#plt.savefig(os.path.join(definitions.OUTPUT_GENERATED_IMAGES_FOLDER, "base_mlp_roc_auc.jpg"))
