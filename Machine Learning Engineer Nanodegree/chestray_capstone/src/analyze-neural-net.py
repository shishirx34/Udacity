import os
import sys 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
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

epochs = sys.argv[1]
attempt = 2
best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.weights.att{}.epochs{}.best.hdf5".format(attempt, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs))

'''
epochs = sys.argv[1]
layer = sys.argv[2]
attempt = sys.argv[3]
best_model_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-transfer-cnn-model.weights.layer{}.att{}.epochs{}.best.hdf5".format(layer, attempt, epochs))
model_testdata_file_path = os.path.join(definitions.SAVED_MODELS_FOLDER, "chestr-transfer-cnn-model.testdata.att{}.epochs{}.npz".format(attempt, epochs))

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
round_pred = pred_Y

for index, label in enumerate(prediction_labels):
    print("Actual:", test_Y[:10,index])
    print("Pred:", round_pred[:10, index])
    #cm = confusion_matrix(test_Y[:,index], round_pred[:, index])
    #print(classification_report(test_Y[:,index], round_pred[:, index], target_names = ['Healthy', label]))
    #plt.matshow(confusion_matrix(test_Y, pred_Y > 0.5))

# Generate the ROC Curve with Area Under the Curve metrics
Visuals.draw_roc_auc("ChestR CNN ROC AUC", prediction_labels, test_Y, pred_Y)
plt.show()
#plt.savefig(os.path.join(definitions.OUTPUT_GENERATED_IMAGES_FOLDER, "base_mlp_roc_auc.jpg"))

sickest_idx = np.argsort(np.sum(test_Y, 1)<1)
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(prediction_labels, 
                                                                  test_Y[idx]) 
                             if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(prediction_labels, 
                                                                  test_Y[idx], pred_Y[idx]) 
                             if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')

plt.show()
#fig.savefig('trained_img_predictions.png')