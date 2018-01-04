# Vectorizer for Bag of Words
import string
import pprint
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

count_vector = CountVectorizer()

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

# Build the token words
token_words = []
for index in documents:
    token_words.append(index.lower().translate(str.maketrans('', '', string.punctuation)).split(' '))

print(token_words)

# Build the counter of the token words
frequency_list = []
for i in token_words:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)

pprint.pprint(frequency_list)

# Using scikit-learn library
count_vector.fit(documents)
docarray = count_vector.transform(documents).toarray()

frequency_matrix = pd.DataFrame(docarray, columns = count_vector.get_feature_names())
print(frequency_matrix)
