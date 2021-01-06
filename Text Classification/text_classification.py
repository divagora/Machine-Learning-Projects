#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 2 14:45:17 2020

@author: ls616
"""

## TEXT CLASSIFICATION ##


## import required dependencies
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from pprint import pprint

from itertools import cycle

from sklearn import metrics, tree
from sklearn import random_projection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import matthews_corrcoef, roc_curve, auc
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from autocorrect import spell

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Embedding, Flattent, MaxPooling1D, Conv1D, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.merge import Concatenate


### load data ###
twenty_train = fetch_20newsgroups(subset="train",shuffle=True)
twenty_test = fetch_20newsgroups(subset="test",shuffle=True)

x_train, x_test = twenty_train.data, twenty_test.data
y_train, y_test = twenty_train.target, twenty_test.target


### inspect data ###
pprint(list(twenty_train.target_names))
print("\n".join(x_train[0].split("\n")))


### preprocess data (optional) ###

# e.g. tokenization, stop words, capitalization, abbreviations, noise removal, 
# spell correction, stemming, lemmatization, word embedding

# n.b. can implement many of these within TfidfVectorizer()



### feature extraction ###
def extract_data(x_train,x_test,max_no_words=100000):
    
    # vectorizer = CountVectorizer(max_features = max_no_words)
    vectorizer = TfidfVectorizer(max_features = max_no_words)
    
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    
    return x_train, x_test

x_train_counts, x_test_counts = extract_data(twenty_train.data,twenty_test.data,10000)



### dimensionality reduction (optional) ###
if False:
    
    ## pca 
    pca = PCA(n_components = 1000)
    x_train_counts_new = pca.fit_transform(x_train_counts)
    x_test_counts_new = pca.transform(x_test_counts)
    
    ## lda
    lda = LinearDiscriminantAnalysis(n_components = 20)
    x_train_counts_new = lda.fit(x_train_counts,y_train)
    x_train_counts_new = lda.transform(x_train_counts)
    x_test_counts_new = lda.transform(x_test_counts)
    
    ## nmf
    nmf = NMF(n_components = 1000)
    x_train_counts_new = nmf.fit(x_train_counts)
    x_train_counts_new = nmf.transform(x_train_counts)
    x_test_counts_new = nmf.transform(x_test_counts)
    
    ## random projection
    random_proj = random_projection.GaussianRandomProjection(n_components=1000)
    x_train_counts_new = random_proj.fit_transform(x_train_counts)
    x_test_counts_new = random_proj.transform(x_test_counts)
    
    ## auto-encoder
    encode_dim = 1000
    
    input = Input(shape=(x_train_counts.shape[1],)) # input placeholder
    encoded = Dense(encode_dim,activation="relu")(input) # encoded representation of input
    decoded = Dense(x_train_counts.shape[1],activation="sigmoid")(encoded) # decoded reconstruction of input
    
    auto_encoder = Model(input,decoded) # map input to reconstructed input
    encoder = Model(input,encoded) # map input to encoded representation
    
    encoded_input = Input(shape=(encode_dim,)) # encoded input
    decoder_layer = auto_encoder.layers[-1] # retrieve final layer of autoencoder
    decoder = Model(encoded_input,decoder_layer(encoded_input)) # create decoder
    
    auto_encoder.compile(optimizer="adadelta",loss="binary_crossentropy") # compile
    auto_encoder.fit(x_train_counts,x_train_counts,epochs=20,batch_size=256,shuffle=True,validation_data=(x_test_counts,x_test_counts),verbose=1)



### modelling ###

## rocchio
model1 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',NearestCentroid()), # classifier
                    ])
                    
model1.fit(x_train,y_train)
model1_preds = model1.predict(x_test) 
print(metrics.classification_report(y_test,model1_preds))



## boosting 
model2 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',GradientBoostingClassifier(n_estimators=50)), # classifier
                    ])

model2.fit(x_train,y_train)
model2_preds = model2.predict(x_test) 
print(metrics.classification_report(y_test,model2_preds))



## bagging 
model3 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',BaggingClassifier(KNeighborsClassifier())), # classifier
                    ])

model3.fit(x_train,y_train)
model3_preds = model3.predict(x_test) 
print(metrics.classification_report(y_test,model3_preds))




## naive Bayes
model4 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',MultinomialNB()), # classifier
                    ])

model4.fit(x_train,y_train)
model4_preds = model4.predict(x_test) 
print(metrics.classification_report(y_test,model4_preds))


# refine model
if False:
    params = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
    gs_model4 = GridSearchCV(model4,params,n_jobs=-1)
    gs_model4 = gs_model4.fit(x_train,y_train)
    
    gs_model4.best_score_
    gs_model4.best_params_




## k-nearest neighbours
model5 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',KNeighborsClassifier()), # classifier
                    ])

model5.fit(x_train,y_train)
model5_preds = model5.predict(x_test) 
print(metrics.classification_report(y_test,model5_preds))




# svm
model6 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',LinearSVC()), # classifier
                    ])

model6.fit(x_train,y_train)
model6_preds = model6.predict(x_test) 
print(metrics.classification_report(y_test,model6_preds))



# decision tree
model7 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',tree.DecisionTreeClassifier()), # classifier
                    ])

model7.fit(x_train,y_train)
model7_preds = model7.predict(x_test) 
print(metrics.classification_report(y_test,model7_preds))


# random forest
model8 = Pipeline([('vect',CountVectorizer()), # vectorizer
                   ('tfidf',TfidfTransformer()), # transformer
                   ('clf',RandomForestClassifier(n_estimators=50)), # classifier
                    ])

model8.fit(x_train,y_train)
model8_preds = model8.predict(x_test) 
print(metrics.classification_report(y_test,model8_preds))


## deep neural net




