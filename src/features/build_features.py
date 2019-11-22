import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import re
import string
import tensorflow as tf
from gensim.models import KeyedVectors

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

class Features():

    def __init__(self, texts_maxlen, headlines_maxlen):

        self.texts_maxlen = texts_maxlen
        self.headlines_maxlen = headlines_maxlen
        self.articles, self.embed_model = self._download_datasets()

        self.headlines, self.texts = self.get_corpus()


    @staticmethod    
    def _download_datasets():

        articles_path = os.path.join(PROJECT_ROOT, "data", "interim", "news-of-the-site-folhauol", "articles.csv")
        embed_path = os.path.join(PROJECT_ROOT, "data", "interim", "glove_s300.txt")
        embed_model_path = os.path.join(PROJECT_ROOT, "data", "interim", "embed_model.pickle")

        if not os.path.isfile(articles_path) or not os.path.isfile(embed_path):
            try:
                subprocess.call([os.path.join(PROJECT_ROOT, "src", "data", "make_dataset.sh")])
                subprocess.call([os.path.join(PROJECT_ROOT, "src", "data", "make_word2vec.sh")])
            except PermissionError:
                print("You need to give executable permission for running ../data/make_dataset.sh and ../data/make_word2vec.sh, \
                    run this command on the terminal: chmod a+x make_dataset.sh at the src/data folder")
                exit()

        if not os.path.isfile(embed_model_path):
            embed_model = KeyedVectors.load_word2vec_format(embed_path)
            with open(embed_model_path, 'wb') as f:
                pickle.dump(embed_model, f)

        else:
            with open(embed_model_path, 'rb') as f:
                embed_model = pickle.load(f)

        return pd.read_csv(articles_path)[['title', 'text']].dropna(), embed_model

    def get_corpus(self):
        '''
        Returns headlines and texts as lists
        '''
        self.articles['title'] = self.articles['title'].apply(lambda x: re.sub(' +', ' ', x.translate(str.maketrans('', '', string.punctuation))).lower())
        self.articles['text'] = self.articles['text'].apply(lambda x: re.sub(' +', ' ', x.translate(str.maketrans('', '', string.punctuation))).lower())

        return self.articles['title'].tolist(), self.articles['text'].tolist()


    def embed_batch(self, batch_texts, batch_headlines):

        batch_embed_texts = np.zeros(shape=(len(batch_texts), self.texts_maxlen, self.embed_model.vector_size))
        batch_embed_headlines = np.zeros(shape=(len(batch_headlines), self.headlines_maxlen, self.embed_model.vector_size))

        for i, batch_text in enumerate(batch_texts):
            for j, word in enumerate(batch_text.split()):
                if j==self.texts_maxlen:
                    break

                if word not in self.embed_model.vocab.keys():
                    batch_embed_texts[i, j, :] = self.embed_model.word_vec('<unk>')
                else:
                    batch_embed_texts[i, j, :] = self.embed_model.word_vec(word)

        for i, batch_headline in enumerate(batch_headlines):
            for j, word in enumerate(batch_headline.split()):
                if j==self.headlines_maxlen:
                    break

                if word not in self.embed_model.vocab.keys():
                    batch_embed_headlines[i, j, :] = self.embed_model.word_vec('<unk>')
                else:
                    batch_embed_headlines[i, j, :] = self.embed_model.word_vec(word)

        return batch_embed_texts, batch_embed_headlines