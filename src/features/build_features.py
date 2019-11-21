import os
from string import punctuation
import subprocess

import nltk
import numpy as np
import pandas as pd
import pickle
import spacy
import tensorflow as tf

import features_helper as hp

data_dir_processed = '../../data/processed'
data_dir_interim = '../../data/interim'

articles_path = os.path.join(
    data_dir_interim, 'news-of-the-site-folhauol/articles.csv')

texts_padded_path = os.path.join(data_dir_processed, "texts-padded")
texts_tokenizer_path = os.path.join(data_dir_processed, "texts-tokenizer.pickle")

headlines_padded_path = os.path.join(data_dir_processed, "headlines-padded")
headlines_tokenizer_path = os.path.join(data_dir_processed, "headlines-tokenizer.pickle")

texts_maxlen = 65
headlines_maxlen = 17
category = 'esporte' #category of news we are going to use


if not os.path.isfile(articles_path):
    try:
        subprocess.call(['../data/make_dataset.sh'])
    except PermissionError:
        print("You need to give executable permission for running ../data/make_dataset.sh, \
            run this command on the terminal: chmod a+x make_dataset.sh at the src/data folder")
        exit()

if (os.path.isfile(texts_padded_path + '.npy') and os.path.isfile(texts_tokenizer_path) and 
    os.path.isfile(headlines_padded_path + '.npy') and os.path.isfile(headlines_tokenizer_path)):
    print("Features files already exist")
    exit()

folha_articles = pd.read_csv(articles_path)

folha_articles = folha_articles[folha_articles['category']==category].dropna()

headlines, texts = folha_articles['title'].tolist(), folha_articles['text'].tolist()

texts_tokenizer, padded_texts = hp.tokenize_and_pad(texts, maxlen=texts_maxlen)
headlines_tokenizer, padded_headlines = hp.tokenize_and_pad(headlines, maxlen=headlines_maxlen)

# saving preprocessed texts
np.save(texts_padded_path, padded_texts)
print("Padded texts stored at: {}".format(texts_padded_path+'.npy'))

# saving texts tokenizer
with open(texts_tokenizer_path, 'wb') as f:
    pickle.dump(texts_tokenizer, f)
print("Texts tokenizer stored at: {}".format(texts_tokenizer_path))

# saving preprocessed headlines
np.save(headlines_padded_path, padded_headlines)
print("Padded headlines stored at: {}".format(headlines_padded_path + '.npy'))

# saving headlines tokenizer
with open(headlines_tokenizer_path, 'wb') as f:
    pickle.dump(headlines_tokenizer, f)
print("Headlines Tokenizer stored at: {}".format(headlines_tokenizer_path))
