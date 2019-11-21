import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from collections import deque
from sklearn.model_selection import train_test_split

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, Tx):
        super(AttentionLayer, self).__init__()
        
        self.repeat = tf.keras.layers.RepeatVector(Tx)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.fc1 = tf.keras.layers.Dense(50, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='tanh')
        self.activate = tf.keras.layers.Softmax(axis=1)
        self.dotor = tf.keras.layers.Dot(axes=1)
        
    def call(self, a, s):
        
        s = self.repeat(s)
        
        x = self.fc2(self.fc1(self.concat([a,s])))
        alphas = self.activate(x)
        context = self.dotor([alphas, a])
        
        return context

class AttentionModel(tf.keras.Model):
    def __init__(self, Tx, Ty, input_size, output_size, embed_size, att_size, hidden_size):
        super(AttentionModel, self).__init__()
        
        self.Ty = Ty
        self.hidden_size = hidden_size
        
        self.embed = tf.keras.layers.Embedding(input_size, embed_size)
        self.attn_layer = AttentionLayer(Tx)
        self.bidir_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=att_size, return_sequences=True))
        self.post_attn_lstm = tf.keras.layers.LSTM(units=hidden_size, return_state=True)
        self.fc = tf.keras.layers.Dense(output_size, activation='softmax')
        
    def call(self, x):
        
        s = tf.zeros(shape=(x.shape[0], self.hidden_size))
        c = tf.zeros(shape=(x.shape[0], self.hidden_size))
        
        x = self.embed(x)
        
        a = self.bidir_lstm(x)
        outputs = []
        
        for t in range(self.Ty):
            context = self.attn_layer(a, s)
            
            s, hid, c = self.post_attn_lstm(context, initial_state=[s,c])
            
            out = self.fc(hid)
            
            outputs.append(out)
            
        return tf.transpose(tf.stack(outputs), [1,0,2])

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def encoded_to_sentence(stn, tokenizer):
    enc = ' '.join([tokenizer.index_word[w] for w in stn if w!=0])
    return enc

def sample_from_model(inputs, target, model, tokenizer, vocab_size):
    idx = np.random.choice(np.arange(len(inputs)), size=1)[0]
    inp = inputs[idx:idx+1]
    tar = target[idx:idx+1]
    
    predictions = model(inp)
    predictions = [np.random.choice(np.arange(vocab_size), p=p.reshape(-1)) for p in predictions[0, ...].numpy()]
    
    print("Original text: {}".format(encoded_to_sentence(inp.reshape(-1).tolist(), tokenizer)))
    print("Original headline: {}".format(encoded_to_sentence(tar.reshape(-1).tolist(), tokenizer)))
    print("Generated headline: {}".format(encoded_to_sentence(predictions, tokenizer)))

def training(X, y, batch_size, att_size, hidden_size, vocab_size, embed_size, 
    epochs, print_every_n=100, sample_every_n=200, learning_rate=1e-3, test_split=0.1, tokenizer=None):

    PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    ckpt_dir = os.path.join(PROJECT_ROOT, 'models', 'ckpts')
    weights_dir = os.path.join(PROJECT_ROOT, 'models', 'weights')

    model = AttentionModel(Tx=X.shape[1], Ty=y.shape[1], input_size=vocab_size, 
        output_size=vocab_size, att_size=att_size, hidden_size=hidden_size, embed_size=embed_size)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    losses = deque(maxlen=print_every_n)

    print("Start training...")

    # training
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for e in range(epochs):
        for inputs, targets in get_batches(X_train, y_train, batch_size):

            ckpt.step.assign_add(1)
            with tf.GradientTape() as tape:
                outputs = model(inputs)
                loss = loss_object(targets, outputs)

            grads = tape.gradient(loss, model.trainable_variables)

            losses.append(loss/inputs.shape[0])

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if int(ckpt.step) % print_every_n == 0:

                ids = np.random.choice(np.arange(len(X_test)), size=batch_size)
                X_test_batch = X_test[ids, ...]
                y_test_batch = y_test[ids, ...]
                test_loss = loss_object(y_test_batch, model(X_test_batch))/batch_size

                print("Epoch: {}/{} ... Avg. Train Loss: {} ... Test Loss for 1 batch: {}".format(e + 1, epochs, np.mean(losses), test_loss))

                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print()


            if int(ckpt.step) % sample_every_n == 0:
                if tokenizer != None:
                    print("\n---Model Sample---")
                    sample_from_model(X_test, y_test, model, tokenizer, vocab_size)
                    print("\n")

        model.save_weights(weights_dir)
        print("\nModel weights stored at: {}\n".format(weights_dir))

