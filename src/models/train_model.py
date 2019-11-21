import numpy as np
import pickle
import os
import tensorflow as tf
from subprocess import call

import models_helper as hp
from attention_model import AttentionModel

data_dir_processed = '../../data/processed'

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
epochs = 30
batch_overfitting_epoch = 3000
batch_size = 8
print_every_n = 100
sample_every_n = 500


model = AttentionModel(Tx, Ty, input_size, output_size, att_size, hidden_size, embed_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './train_ckpts', max_to_keep=3)

losses = []

print("Start training...")

# training
with tf.device('/gpu:0'):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for e in range(epochs):
        for inputs, targets in hp.get_batches(texts_padded, headlines_padded, batch_size):

            ckpt.step.assign_add(1)
            with tf.GradientTape() as tape:
                outputs = model(inputs)
                loss = loss_object(targets, outputs)

            grads = tape.gradient(loss, model.trainable_variables)

            losses.append(loss/inputs.shape[0])

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if int(ckpt.step) % print_every_n == 0:
                print("Epoch: {}/{} ... Avg. Loss: {}".format(e + 1, epochs, np.mean(losses[-print_every_n:])))

                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if int(ckpt.step) % sample_every_n == 0:
                print("\n---Model Sample---\n")
                hp.sample_from_model(texts_padded, headlines_padded, model, texts_tokenizer, headlines_tokenizer)
                print("\n\n")
