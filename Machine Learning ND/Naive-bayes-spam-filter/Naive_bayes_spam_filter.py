# Main file to read data and define classifier

import pandas as pd

df = pd.read_table('./dataset/SMSSpamCollection',
    sep = '\t',
    header = None,
    names = ['label', 'message'])

#Map labels to binary values
df['label'] = df.label.map({'ham': 0, 'spam': 1})

#Print out first 5 rows
print(df.shape)
print(df.head())