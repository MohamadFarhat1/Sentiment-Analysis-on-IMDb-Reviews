import pandas as pd
from Doc2Vec import DocIterator
import gensim as gen
import numpy as np
import os
import itertools

# Read files and store into dataframe

# First, read labeled data
# Create empty lists
reviewtext = []
uniqueid = []
dataset = []
sentiment = []
# Training or test folders
for train_test in ['train', 'test']:
    # Negative or Positive folders
    for neg_pos in ['neg', 'pos']:
        # Each text file
        for tf in os.listdir('C:/Blog/ab/' + train_test + '/' + neg_pos):
            text = open('C:/Blog/ab/' + train_test + '/' + neg_pos + '/' + tf, "r", encoding='utf-8')
            reviewtext.append(text.readlines())
            uniqueid.append(tf)
            dataset.append(train_test)
            sentiment.append(neg_pos)

# Then, read unlabeled data (to use in doc2vec training only)
for tf in os.listdir('C:/Blog/ab/train/unsup'):
    text = open('C:/Blog/ab/train/unsup/' + tf, "r", encoding='utf-8')
    reviewtext.append(text.readlines())
    uniqueid.append(tf)
    dataset.append('unsup')
    sentiment.append('null')

# Convert the nested list of reviews to a flat list
reviewtext = list(itertools.chain.from_iterable(reviewtext))

# Create a dataframe
reviews = pd.DataFrame({'Id': pd.Series(uniqueid),
                        'Dataset': pd.Series(dataset),
                        'Sentiment': pd.Series(sentiment),
                        'Review': pd.Series(reviewtext)})
# Create unique Ids
reviews.index = range(reviews.shape[0])
for i in range(reviews.shape[0]):
    reviews.loc[i, 'Id'] = 'r' + str(i)

# Remove break tags (that I noticed in the text files)
reviews['Review'] = reviews['Review'].map(lambda s: s.replace('<br />', ' '))

# Create lists of reviews (as documents), and their IDs (as labels)
docs = list(reviews['Review'])
labels = list(reviews['Id'])

# Create a Doc2Vec class instance
d2v = DocIterator(docs, labels)
# Make sure labels are unique
d2v.checkLabelsUnique()
# Pre-process documents before training
# This includes fixing encoding, normalizing text, remove punctuation/symbols, convert to lowercase, detect phrases, etc
d2v.prepareDocs()

# Initialize doc2vec model
model = gen.models.Doc2Vec(size=300, window=10, min_count=1, dm=0, workers=8, alpha=0.025, min_alpha=0.025, negative=5,
                           hs=0)

# Build vocabulary
model.build_vocab(d2v)

# Do model training over several epochs, shuffle documents each time, decrease alpha each time
for epoch in range(10):
    print('Epoch %d of 10.' % (epoch + 1))
    d2v.shuffleDocs()
    model.train(d2v)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
print('Done!')

# Save model
model.save('C:/Blog/ab/d2v_model_reviews')

# Load model
model = gen.models.Doc2Vec.load('C:/Blog/ab/d2v_model_reviews')

# Create array to document labels and corresponding document vectors
docvecs = np.zeros(shape=(np.shape(model.docvecs)[0], np.shape(model.docvecs)[1] + 1))
labels = []
# Populate this array
for i in range(np.shape(docvecs)[0]):
    labels.append(model.docvecs.index_to_doctag(i))
    docvecs[i, 1:(np.shape(docvecs)[1])] = model.docvecs[model.docvecs.index_to_doctag(i)]
# Convert to dataframe
docvecs = pd.DataFrame(docvecs)
# Add labels (use this to check if matching with labels in dataframe when concatenating later)
docvecs[0] = pd.Series(labels)

# Concatenate with the original dataframe containing text
docvecs = pd.concat([reviews, docvecs], axis=1)

# Save to file
docvecs.to_csv('C:/Blog/ab/docvecs.csv', header=1, index=0)
