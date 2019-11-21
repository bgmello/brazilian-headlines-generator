import numpy as np
import tensorflow as tf

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def encoded_to_sentence(stn, tokenizer):
    return ' '.join([tokenizer.index_word[w] for w in stn if w!=0])

def sample_from_model(inputs, target, model, texts_tokenizer, headlines_tokenizer):
    idx = np.random.choice(np.arange(len(inputs)), size=1)[0]
    inp = inputs[idx:idx+1]
    tar = target[idx:idx+1]
    
    predictions = model(inp)
    predictions = [np.random.choice(np.arange(len(headlines_tokenizer.word_counts)+1), p=p.reshape(-1)) for p in predictions[0, ...].numpy()]
    
    print("Original text: {}".format(encoded_to_sentence(inp.reshape(-1).tolist(), texts_tokenizer)))
    print("Original headline: {}".format(encoded_to_sentence(tar.reshape(-1).tolist(), headlines_tokenizer)))
    print("Generated headline: {}".format(encoded_to_sentence(predictions, headlines_tokenizer)))