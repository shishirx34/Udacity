# Spam filter - classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_table('./dataset/SMSSpamCollection',
    sep = '\t',
    header = None,
    names = ['label', 'message'])

# Map labels to binary values
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Split the data for training and testing
msg_train, msg_test, label_train, label_test = train_test_split(df['message'],
                                                                df['label'],
                                                                random_state = 1)

# Format the training data into bag of words
count_vector = CountVectorizer()

msg_training_matrix = count_vector.fit_transform(msg_train)
msg_testing_matrix = count_vector.transform(msg_test)

print('Total set: ' + str(df.shape[0]))
print('Training set size: ' + str(msg_train.shape[0]))
print('Test set size: ' + str(msg_test.shape[0]))

# Classify data with naive bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(msg_training_matrix, label_train)

# Make predictions on test data
predictions = naive_bayes.predict(msg_testing_matrix)

# Test the predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: ' + str(accuracy_score(label_test, predictions)))
print('Precision: ' + str(precision_score(label_test, predictions)))
print('Recall(sensitivity): ' + str(recall_score(label_test, predictions)))
print('F1 Score: ' + str(f1_score(label_test, predictions)))
