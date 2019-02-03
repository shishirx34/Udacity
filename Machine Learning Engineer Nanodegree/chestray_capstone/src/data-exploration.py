# Feature exploration for ChestR

import numpy as np 
import pandas as pd 
import os 
from Configurations.Config import Config
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from definitions import ROOT_DIR

sns.set_style('whitegrid')

OUTPUT_GENERATED_IMAGES_FOLDER = os.path.join(ROOT_DIR, "Generated", "data-exploration-images")

# Load Configs
config = Config.GetConfig()

# Load Data files
data_file = os.path.join(config.DataFolderPath, "Data_Entry_2017.csv")
df = pd.read_csv(data_file)

# Data load check
print(df.head())

#Explore input data features
print("\nTotal #XRay Reports:", df.shape[0])

#Get all disease labels only
all_labels = df['Finding Labels'].str.split("|", n=None, expand=True).stack().reset_index()[0].unique()
disease_labels = np.delete(all_labels, np.where(all_labels=='No Finding'), axis=0)
print("Total #Disease labels in dataset:", disease_labels.shape[0])

#Drop unrelated data columns
df = df[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position']]

#Co-relate all labels with data
for label in all_labels:
    df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)

def plot_distribution(group_by_col, distribution_list, group_data, savetofile):
    plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(8,1)

    ax1 = plt.subplot(gs[:7, :])
    ax1.set(ylabel="",xlabel="#X-Rays with Label")
    ax1.legend(fontsize=20)
    ax1.set_title('X-Ray Labels distribution',fontsize=18)

    data1 = pd.melt(df,
                id_vars=[group_by_col],
                value_vars = list(distribution_list),
                var_name = 'Category',
                value_name = 'Count')
    data1 = data1.loc[data1.Count>0]

    if group_data == True:
        g=sns.countplot(x='Category',hue=group_by_col,data=data1, ax=ax1, order = data1['Category'].value_counts().index)
    else:
        g=sns.countplot(x='Category',data=data1, ax=ax1, order = data1['Category'].value_counts().index)

    plt.subplots_adjust(hspace=0.5)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(OUTPUT_GENERATED_IMAGES_FOLDER, savetofile))

#Plot Finding Labels distribution
print("\nGenerating XRay Reports vs Labels distribution plot...") 
plot_distribution(group_by_col='Patient Gender', distribution_list=all_labels, group_data=False, savetofile="labels_distribution.jpg")

#Plot patient age distribution
print("\nGenerating Xray reports vs Patient Age distribution plot...")
g = sns.catplot(x="Patient Age", hue="Patient Gender",data=df, kind="count",height=10)
g.set_xticklabels(np.arange(0,100))
g.set_xticklabels(step=10)
g.set(ylabel="# of X-Rays")
g.fig.suptitle('Age distribution by Gender',fontsize=18)
plt.savefig(os.path.join(OUTPUT_GENERATED_IMAGES_FOLDER, "age_gender_distribution.jpg"))

#Plot Labels vs Gender distribution
print("\nGenerating XRay report Labels vs Patient Gender distribution plot...") 
plot_distribution(group_by_col='Patient Gender', distribution_list=all_labels, group_data=True, savetofile="labels_vs_gender.jpg")

#Plot Labels vs Age vs Gender distribution
print("\nGenerating Xray report Labels vs Patient Age distribution plot...")
f, axarr = plt.subplots(7, 2, sharex=True,figsize=(15, 20))
i=0
j=0
x=np.arange(0,100,10)
for disease in disease_labels:
    g=sns.countplot(x='Patient Age', hue="Patient Gender",data=df[df['Finding Labels']==disease], ax=axarr[i, j])
    axarr[i, j].set_title(disease)
    g.set_xlim(0,90)
    g.set_xticks(x)
    g.set_xticklabels(x)
    j=(j+1)%2
    if j==0:
        i=(i+1)%7
f.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(OUTPUT_GENERATED_IMAGES_FOLDER, "disease_age_gender_distribution.jpg"))

#Plot MultiLabel relations
print("\nGenerating multi label co-relation plot...")
df2 = df[disease_labels]
df2['Total'] = df2.sum(axis=1)

I = pd.Index(["Single Disease", "Multiple Diseases"], name="rows")
C = pd.Index(disease_labels, name="columns")

'''
print(df2.head())
print(multi_disease_df.head())

for index, row in df2.iterrows():
    if (row["Total"] == 1):
        update_index = "Single Disease"
    else:
        update_index = "Multiple Diseases"

    for label in disease_labels:
        multi_disease_df.ix[update_index, label] += row[label]

'''

# Generated from above commented code
generated_disease_data = [[1093, 892, 3955, 110, 9547, 2139, 2705, 4215, 2194, 1126, 322, 727, 628, 1310],
[1683, 1624, 9362, 117, 10347, 3643, 3626, 7344, 3108, 2259, 1109, 959, 1675, 3357]]
multi_disease_df = pd.DataFrame(generated_disease_data, index=I, columns=C)

print(multi_disease_df.head())
multi_disease_df = multi_disease_df.T
multi_disease_df = multi_disease_df.sort_values('Multiple Diseases', ascending=False)
ax = multi_disease_df[['Single Disease', 'Multiple Diseases']].plot(kind='bar', title ="Multi disease distribution", figsize=(15, 10), legend=True, fontsize=12)
plt.savefig(os.path.join(OUTPUT_GENERATED_IMAGES_FOLDER, "multiple_disease_distribution.jpg"))
