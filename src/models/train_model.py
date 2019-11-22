import numpy as np
import os
import pickle
import tensorflow as tf
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, Tx):
        super(AttentionLayer, self).__init__()
        
        self.repeat = tf.keras.layers.RepeatVector(Tx)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.fc1 = tf.keras.layers.Dense(200, activation='relu')
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
    def __init__(self, Tx, Ty, input_size, output_size, att_size, hidden_size):
        super(AttentionModel, self).__init__()
        
        self.Ty = Ty
        self.hidden_size = hidden_size
        self.attn_layer = AttentionLayer(Tx)
        self.bidir_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=att_size, return_sequences=True))
        self.post_attn_lstm = tf.keras.layers.LSTM(units=hidden_size, return_state=True)
        self.fc = tf.keras.layers.Dense(output_size, activation='softmax')
        
    def call(self, x):
        
        s = tf.zeros(shape=(x.shape[0], self.hidden_size))
        c = tf.zeros(shape=(x.shape[0], self.hidden_size))
        
        
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

def sample_from_model(inputs, target, model, features):
    idx = np.random.choice(np.arange(len(inputs)), size=1)[0] 
    inp, tar = features.embed_batch(inputs[idx:idx+1], target[idx:idx+1])
    predictions = model(inp).numpy()[0]
    pred = ' '.join([features.embed_model.similar_by_vector(p.reshape((300,)), topn=1)[0][0] for p in predictions])
    
    print("Original text: {}".format(' '.join(inputs[idx].split()[:features.headlines_maxlen])))
    print("Original headline: {}".format(' '.join(target[idx].split()[:features.texts_maxlen])))
    print("Generated headline: {}".format(pred))

@tf.function
def train_step(inputs, targets, model, optimizer, train_loss):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.keras.losses.mean_squared_error(targets, outputs)

    grads = tape.gradient(loss, model.trainable_variables)

    train_loss(loss)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def training(X, y, batch_size, att_size, hidden_size, input_size, output_size, 
    epochs, features, print_every_n=100, sample_every_n=500, learning_rate=1e-3, test_split=0.1):


    PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

    ckpt_dir = os.path.join(PROJECT_ROOT, 'models', 'ckpts')
    weights_path = os.path.join(PROJECT_ROOT, 'models', 'weights-attention.hdf5')

    model = AttentionModel(Tx=features.texts_maxlen, Ty=features.headlines_maxlen, input_size=input_size, 
        output_size=output_size, att_size=att_size, hidden_size=hidden_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    print("Start training...")

    # training
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for e in range(epochs):
        
        train_loss.reset_states()
        test_loss.reset_states()

        for inputs, targets in get_batches(X_train, y_train, batch_size):

            ckpt.step.assign_add(1)

            inputs, targets = features.embed_batch(inputs, targets)

            train_step(inputs, targets, model, optimizer, train_loss)

            if int(ckpt.step) % print_every_n == 0:

                ids = np.random.choice(np.arange(len(X_test)-batch_size), size=1)[0]
                X_test_batch = X_test[ids:ids+batch_size]
                y_test_batch = y_test[ids:ids+batch_size]
                
                X_test_batch, y_test_batch = features.embed_batch(X_test_batch, y_test_batch)

                test_loss(tf.keras.losses.mean_squared_error(y_test_batch, model(X_test_batch)))

                print("Epoch: {}/{} ... Avg. Train Loss: {} ... Test Loss for 1 batch: {}".format(e + 1, epochs, train_loss.result(),test_loss.result()))

                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print()


            if int(ckpt.step) % sample_every_n == 0:
                print("\n---Model Sample---")
                sample_from_model(X_test, y_test, model, features)
                print("\n")

        model.save_weights(weights_path)
        print("\nModel weights stored at: {}\n".format(weights_path))

