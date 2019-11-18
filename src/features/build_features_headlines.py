import pandas as pd
import numpy as np
import nltk

import tensorflow as tf

from string import punctuation
import pickle
import os

data_dir_processed = '../../data/processed'
data_dir_interim = '../../data/interim'

folha_articles = pd.read_csv(os.path.join(data_dir_interim, 'news-of-the-site-folhauol/articles.csv'))

headlines = folha_articles['title'].tolist()

headlines_tokenizer = tf.keras.preprocessing.text.Tokenizer()
headlines_tokenizer.fit_on_texts(headlines)

encoded_headlines = headlines_tokenizer.texts_to_sequences(headlines)

padded_headlines = tf.keras.preprocessing.sequence.pad_sequences(encoded_headlines, maxlen=16, 
                                                                 padding='post', truncating='post')

#saving preprocessed headlines
np.save(os.path.join(data_dir_processed, "headlines-padded"), padded_headlines)

print("Padded headlines stored at: {}".format(os.path.join(data_dir_processed, "headlines-padded.npy")))

with open(os.path.join(data_dir_processed, "headlines-tokenizer.pickle"), 'wb') as f:
    pickle.dump(headlines_tokenizer, f)

print("Headlines Tokenizer stored at: {}".format(os.path.join(data_dir_processed, "headlines-tokenizer.pickle")))