# Feature exploration for ChestR

import numpy as np 
import pandas as pd 
import os 
from Configurations.Config import Config

# Load Configs
config = Config.GetConfig()

# Load Data files
data_file = os.path.join(config.DataFolderPath, "Data_Entry_2017.csv")
df = pd.read_csv(data_file)

# Data load check
print(df.head())

#Explore input data features
print("\nTotal #XRay Reports:", df.shape[0])

#Get all disease labels
disease_labels = df['Finding Labels'].str.split("|", n=None, expand=True).stack().reset_index()[0].unique()
print("Total #Finding labels in dataset:", disease_labels.shape[0])

#Plot Finding Labels distribution

#Plot Gender distribution
print("\nGenerating XRay Reports vs Patient Gender distribution plot...") 
print(df['Patient Gender'].value_counts())

#Plot patient age distribution
print("\nGenerating Xray reports vs Patient Age distribution plot...")

#Plot Labels vs Gender distribution
print("\nGenerating XRay report Labels vs Patient Gender distribution plot...") 

#Plot Labels vs Age
print("\nGenerating Xray report Labels vs Patient Age distribution plot...")

#Plot MultiLabel relations
print("\nGenerating multi label co-relation plot...")