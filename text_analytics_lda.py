# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:40:26 2015

@author: GHT438
"""

import numpy as np
import lda
import lda.datasets
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data_df=pd.read_csv('Test_Survey.csv')
data_df.Verbatim=data_df.Verbatim.fillna('and')

document=[]
for line in data_df.Verbatim:
    document.append(line.lower())

frequency=[]
for words in document:
    for token in str(words).split():
        frequency.append(token)

counts = Counter(frequency)

words_list=[]
for key,value in counts.iteritems():
    if value > 1:
        words_list.append(key)
    
vectorizer=CountVectorizer(input=document,stop_words='english',strip_accents='ascii',decode_error = 'ignore')
dtm = vectorizer.fit_transform(document)
vocab = vectorizer.get_feature_names()

dtm = dtm.toarray() 
vocab = np.array(vocab)
model = lda.LDA(n_topics=20, n_iter=150, random_state=1)
model.fit(dtm)
topic_word = model.topic_word_
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = vocab[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))