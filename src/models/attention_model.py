import tensorflow as tf

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
    def __init__(self, Tx, Ty, input_size, target_size, embed_size, att_size, hidden_size):
        super(AttentionModel, self).__init__()
        
        self.Ty = Ty
        self.hidden_size = hidden_size
        
        self.embed = tf.keras.layers.Embedding(input_size, embed_size)
        self.attn_layer = AttentionLayer(Tx)
        self.bidir_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=att_size, return_sequences=True))
        self.post_attn_lstm = tf.keras.layers.LSTM(units=hidden_size, return_state=True)
        self.fc = tf.keras.layers.Dense(target_size, activation='softmax')
        
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