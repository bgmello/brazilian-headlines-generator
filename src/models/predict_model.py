import numpy as np
import pickle
import os
import tensorflow as tf
from subprocess import call

import models_helper as hp
from attention_model import AttentionModel

data_dir_processed = '../../data/processed'
models_train_dir = '../../models/train_ckpts'
model_weights_dir = '../../models/attention-model.hdf5'

texts_padded_path = os.path.join(data_dir_processed, "texts-padded.npy")
texts_tokenizer_path = os.path.join(data_dir_processed, "texts-tokenizer.pickle")

headlines_padded_path = os.path.join(data_dir_processed, "headlines-padded.npy")
headlines_tokenizer_path = os.path.join(data_dir_processed, "headlines-tokenizer.pickle")


try:
    with open(headlines_tokenizer_path, 'rb') as f:
        headlines_tokenizer = pickle.load(f)

    headlines_padded = np.load(headlines_padded_path)

    with open(texts_tokenizer_path, 'rb') as f:
        texts_tokenizer = pickle.load(f)

    texts_padded = np.load(texts_padded_path)

except FileNotFoundError:
    call(["python", "../features/build_features.py"])

# hyperparameters
att_size = 128
hidden_size = 256
Tx = texts_padded.shape[1]
Ty = headlines_padded.shape[1]
input_size = len(texts_tokenizer.word_counts)+1
output_size = len(headlines_tokenizer.word_counts)+1
embed_size = 300
epochs = 0
batch_overfitting_epoch = 3000
batch_size = 8
print_every_n = 100
sample_every_n = 500


model = AttentionModel(Tx, Ty, input_size, output_size, att_size, hidden_size, embed_size)

model.load_weights(model_weights_dir)

def generate_headline_from_string(stn, model):
	'''
	Generates a headline using the model
	'''

	encoded_stn = texts_tokenizer.texts_to_sequences([stn])
	padded_stn = tf.keras.preprocessing.sequence.pad_sequences(encoded_stn, maxlen=65, padding='post', truncating='post')
	predictions = model(padded_stn)
	predictions = tf.argmax(predictions, axis=-1).numpy()[0].tolist()
	print(predictions)
	# predictions = [np.random.choice(np.arange(len(headlines_tokenizer.word_counts)+1), p=p.reshape(-1)) for p in predictions[0, ...].numpy()]
	print("Generated headline: {}".format(hp.encoded_to_sentence(predictions, headlines_tokenizer)))


if __name__=='__main__':
	generate_headline_from_string('Ex-treinador das equipes olímpicas de hipismo do Estados Unidos e do Brasil, George Morris, foi excluído permanentemente do esporte depois de uma investigação sobre “delitos de conduta sexual envolvendo menores de idade”', model)