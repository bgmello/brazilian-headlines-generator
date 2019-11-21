from features.build_features import Features
from models.train_model import training

#hyperparameters
texts_maxlen = 65
headlines_maxlen = 17
num_words = 40000
category = 'esporte'
batch_size = 8
att_size = 32
hidden_size = 256
embed_size = 300
epochs = 30

features = Features(texts_maxlen, headlines_maxlen, 
	num_words)

training(X=features.padded_texts, y=features.padded_headlines, 
	att_size=att_size, hidden_size=hidden_size, vocab_size=num_words, 
	embed_size=embed_size, tokenizer=features.tokenizer,
	epochs=epochs, batch_size=batch_size)