import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

class Features():

    def __init__(self, texts_maxlen, headlines_maxlen, num_words):

        self.texts_maxlen = texts_maxlen
        self.headlines_maxlen = headlines_maxlen
        self.num_words = num_words
        self.articles = self._download_dataset()

        self.headlines, self.texts = self.get_corpus()

        self.build_features()


    @staticmethod    
    def _download_dataset():

        articles_path = os.path.join(PROJECT_ROOT, "data", "interim", "news-of-the-site-folhauol", "articles.csv")

        if not os.path.isfile(articles_path):
            try:
                subprocess.call(['../data/make_dataset.sh'])
            except PermissionError:
                print("You need to give executable permission for running ../data/make_dataset.sh, \
                    run this command on the terminal: chmod a+x make_dataset.sh at the src/data folder")
                exit()

        return pd.read_csv(articles_path).dropna()

    def build_features(self):

        data_dir_processed = os.path.join(PROJECT_ROOT, "data", "processed")

        texts_padded_path = os.path.join(data_dir_processed, "texts-padded")
        tokenizer_path = os.path.join(data_dir_processed, "tokenizer.pickle")
        headlines_padded_path = os.path.join(data_dir_processed, "headlines-padded")

        if os.path.isfile(headlines_padded_path) and os.path.isfile(texts_padded_path) and os.path.isfile(tokenizer_path):

            self.padded_headlines = np.load(headlines_padded_path)

            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

            self.padded_texts = np.load(texts_padded_path)

        else:
            self.tokenize_and_pad()
            self.save_tokenizer_and_padded()

    def get_corpus(self):
        '''
        Returns headlines and texts as lists
        '''
        return self.articles['title'].tolist(), self.articles['text'].tolist()

    def tokenize_and_pad(self, padding='post', truncating='post'):
        '''
        Creates keras tokenizer for input_corpus and output_corpus and returns padded sequences

        Args:
                padding(string): 'post' or 'pre' type of padding to be used
                truncating(string): 'post' or 'pre' type of truncating to be used
        '''

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(self.texts + self.headlines)

        encoded_input_corpus, encoded_output_corpus = tokenizer.texts_to_sequences(self.texts), tokenizer.texts_to_sequences(self.headlines)

        padded_input_corpus = tf.keras.preprocessing.sequence.pad_sequences(encoded_input_corpus, maxlen=self.texts_maxlen, padding=padding, truncating=truncating)
        padded_output_corpus = tf.keras.preprocessing.sequence.pad_sequences(encoded_output_corpus, maxlen=self.headlines_maxlen, padding=padding, truncating=truncating)

        self.tokenizer = tokenizer
        self.padded_texts = padded_input_corpus
        self.padded_headlines = padded_output_corpus

    def save_tokenizer_and_padded(self):

        data_dir_processed = os.path.join(PROJECT_ROOT, "data", "processed")

        texts_padded_path = os.path.join(data_dir_processed, "texts-padded")
        tokenizer_path = os.path.join(data_dir_processed, "tokenizer.pickle")
        headlines_padded_path = os.path.join(data_dir_processed, "headlines-padded")

        # saving texts tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # saving preprocessed texts
        np.save(texts_padded_path, self.padded_texts)

        # saving preprocessed headlines
        np.save(headlines_padded_path, self.padded_headlines)

        