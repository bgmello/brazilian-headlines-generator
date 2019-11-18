import pandas as pd
import numpy as np
import nltk
import spacy

import tensorflow as tf

from string import punctuation
import pickle
import os

data_dir_processed = '../../data/processed'
data_dir_interim = '../../data/interim'

folha_articles = pd.read_csv(os.path.join(data_dir_interim, 'news-of-the-site-folhauol/articles.csv'))

texts = folha_articles['text'].tolist()

def drop_stopwords(texts):
    '''
    Remove portuguese stopwords from corpus
    
    Args:
        texts: corpus
    
    Returns:
        Corpus without the stopwords
    '''
    new_texts = []
    stop_words = nltk.snowball.stopwords.words('portuguese')
    for stn in texts:
        if pd.isna(stn):
            new_texts.append('')
        else:
            new_texts.append(' '.join([word for word in stn.split() 
                                       if word.lower() not in stop_words]))
    return new_texts

def preprocess_string(stn):
    '''
    Remove punctuation and lower case sentence
    '''
    if pd.isna(stn):
        return ''
    stn = stn.lower()
    return ''.join([c for c in stn if c not in punctuation])

def lemmatize_string(stn, lemmatizer):
    '''
    Lemmatize words in sentence
    '''
    new_stn = []
    for token in lemmatizer(stn):
        new_stn.append(token.lemma_)
        
    return ' '.join(new_stn)


texts = [preprocess_string(stn) for stn in texts]
texts = drop_stopwords(texts)

lemmatizer = spacy.load("pt_core_news_sm")

try:
    print("Lemmatized texts restored from {}".format(os.path.join(data_dir_processed, "lemmatized-texts.pickle")))
    with open(os.path.join(data_dir_processed, "lemmatized-texts.pickle"), 'rb') as f:
        lemmatized_texts = pickle.load(f)
    
except:
    lemmatized_texts = []


start = len(lemmatized_texts)
end = len(texts)

for i in range(start, end):
    stn = texts[i]
    lemmatized_texts.append(lemmatize_string(stn, lemmatizer))
    if i%1000==0:
        with open(os.path.join(data_dir_processed, "lemmatized-texts.pickle"), 'wb') as f:
            pickle.dump(lemmatized_texts, f)
            
        print("Lemmatized texts saved at {} ... itr: {}".format(os.path.join(data_dir_processed, "lemmatized-texts.pickle"), i))



texts_tokenizer = tf.keras.preprocessing.text.Tokenizer()
texts_tokenizer.fit_on_texts(lemmatized_texts)

encoded_texts = texts_tokenizer.texts_to_sequences(lemmatized_texts)
padded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, maxlen=1000,
                                                             truncating='post', padding='post')

#saving preprocessed texts
np.save(os.path.join(data_dir_processed, "texts-padded"), padded_texts)

print("Padded texts stored at: {}".format(os.path.join(data_dir_processed, "texts-padded.npy")))

with open(os.path.join(data_dir_processed, "texts-tokenizer.pickle"), 'wb') as f:
    pickle.dump(texts_tokenizer, f)

print("Texts tokenizer stored at: {}".format((os.path.join(data_dir_processed, "texts-tokenizer.pickle"))))
