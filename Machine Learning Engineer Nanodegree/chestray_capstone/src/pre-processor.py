import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from glob import glob
from Configurations.Config import Config
from definitions import ROOT_DIR
import itertools
from sklearn.model_selection import train_test_split
from Utilities.MeasureDuration import MeasureDuration
import time
from Utilities.TensorLoader import Load_Data_Images_For_DataFrame

OUTPUT_GENERATED_IMAGES_FOLDER = os.path.join(ROOT_DIR, "Generated", "data-exploration-images")

def PlotSampledLabels():
    label_counts = xray_df['Finding Labels'].value_counts()[:15]
    fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
    ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts))+0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
    ax1.set_title("Sampled Label distribution")

# Load Configs
config = Config.GetConfig()
 
# Load Data files
with MeasureDuration() as m: 
      print("Loading data...")
      data_file = os.path.join(config.DataFolderPath, "Data_Entry_2017.csv")
      xray_df = pd.read_csv(data_file)
      all_images_map = {os.path.basename(x): x for x in glob(os.path.join(config.DataFolderPath, 'images*', '*', '*.png'))}
      print("Total data loaded: ", xray_df.shape[0])
      print("Loaded paths for images: ", len(all_images_map))

# Insert image path into data
xray_df['Image Path'] = xray_df['Image Index'].map(all_images_map.get)

#Get all disease labels only
all_labels = xray_df['Finding Labels'].str.split("|", n=None, expand=True).stack().reset_index()[0].unique()
disease_labels = np.delete(all_labels, np.where(all_labels=='No Finding'), axis=0)
print("Total #Disease labels in dataset:", disease_labels.shape[0])

#Drop unrelated data columns
xray_df = xray_df[['Image Index', 'Finding Labels', 'Patient ID', 'Image Path']]

#Co-relate all disease labels with data => adding probability of 1.0 for disease labels
for label in disease_labels:
    xray_df[label] = xray_df['Finding Labels'].apply(lambda x: 1.0 if label in x else 0.0)

print('Disease Labels distribution ({})'.format(len(disease_labels)), 
      [(label,int(xray_df[label].sum())) for label in disease_labels])

# Clean up a category with less examples given the size distribution.
disease_labels = [label for label in disease_labels if xray_df[label].sum() > 500]
print('Cleaned up disease Labels distribution ({})'.format(len(disease_labels)), 
      [(label,int(xray_df[label].sum())) for label in disease_labels])

# Given the nature of the data skewed towards "No Finding", we can reduce the data size to something reasonable.
# Generate sample weights as a function of all the disease labels in it.
sample_weights = xray_df['Finding Labels'].map(lambda x: len(x.split('|')) 
    if len(x) > 0 and x not in 'No Finding' else 0).values + 15e-2
xray_df = xray_df.sample(50000, weights=sample_weights)
print("Dataset shape after sampling:", xray_df.shape)

# Generate plot for the sampled data
# PlotSampledLabels()
# plt.savefig(os.path.join(OUTPUT_GENERATED_IMAGES_FOLDER, "sampled_disease_distribution.jpg"))

# Create disease prediction matrix from all the disease labels we have
with MeasureDuration() as m:
      print("Generating disease prediction vector...")
      xray_df['Disease Prediction'] = xray_df.apply(lambda x: [x[disease_labels].values], 1).map(lambda x: x[0])
      #print(xray_df.sample(10))

# Split the data into Training set and Testing set.
# 20% of the data is for testing
# Keep this data consitent in training and testing different models.
with MeasureDuration() as m:
      print("Splitting the data for train and test...")
      train_xray_df, val_test_xray_df = train_test_split(xray_df, test_size=0.20, random_state=42)
      val_xray_df, test_xray_df = train_test_split(val_test_xray_df, test_size=0.40, random_state=42)
      print("Training set size:", train_xray_df.shape)
      print("Validation set size:", val_xray_df.shape)
      print("Testing set size:", test_xray_df.shape)

with MeasureDuration() as m:
      print("Load images and generate dataframes...")
      train_gen_df = Load_Data_Images_For_DataFrame(
            train_xray_df,
            path_column='Image Path',
            prediction_column='Disease Prediction',
            batch_size=40
      )

      val_gen_df = Load_Data_Images_For_DataFrame(
            val_xray_df,
            path_column='Image Path',
            prediction_column='Disease Prediction',
            batch_size=200
      )

      test_gen_df = Load_Data_Images_For_DataFrame(
            test_xray_df,
            path_column='Image Path',
            prediction_column='Disease Prediction',
            batch_size=1000
      )

print("Data load complete!")