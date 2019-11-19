import tensorflow as tf

def get_first_paragraph(stn):
    '''
    Returns first paragraph of stn(string) as a list of words
    '''
    try:
        return stn.split('  ')[0].split(' ')

    except:
        return ['']


def tokenize_and_pad(corpus, maxlen, padding='post', truncating='post'):
    '''
    Creates keras tokenizer for corpus and returns padded sequence

    Args:
            corpus(list of strings): Corpus to be use to create the tokenizer
            maxlen(int): Length to pad and trunacate corpus
            padding(string): 'post' or 'pre' type of padding to be used
            truncating(string): 'post' or 'pre' type of truncating to be used

    Returns:
            tokenizer(tf.keras.preprocessing.text.Tokenizer): Tokenizer fitted on corpus 
            padded_corpus(np.array): Padded corpus
    '''

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(corpus)

    encoded_corpus = tokenizer.texts_to_sequences(corpus)

    padded_corpus = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_corpus, maxlen=maxlen, padding=padding, truncating=truncating)

    return tokenizer, padded_corpus