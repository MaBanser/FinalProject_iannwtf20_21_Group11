import utils

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

class MultiHeadAttention(Layer):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        
        assert model_dim % num_heads == 0

        self.head_dim = model_dim // num_heads

        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)

        self.wo = tf.keras.layers.Dense(model_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask, **kwargs):
        batch_size = x.shape[0]

        query = self.wq(x)  # (batch_size, seq_len, model_dim)
        if 'encoder_out' in kwargs:
            key = self.wk(kwargs['encoder_out'])
            value = self.wv(kwargs['encoder_out'])
        else:
            key = self.wk(x)
            value = self.wv(x)

        # Split up the heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # attention.shape == (batch_size, num_heads, seq_len, head_dim)
        attention, attention_weights = utils.scaled_dot_product_attention(query, key, value, mask)

        # Concatenate heads back together
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim)
        attention = tf.reshape(attention,(batch_size, -1, self.model_dim))  # (batch_size, seq_len, model_dim)

        output = self.wo(attention)  # (batch_size, seq_len, output_dim)

        return output, attention_weights

class EncoderLayer(Layer):
    def __init__(self, model_dim, num_heads, ffn_units, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()

        self.feedForward = [
            tf.keras.layers.Dense(ffn_units, activation='relu'),
            tf.keras.layers.Dense(model_dim)
        ]
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x, mask, training = False):
        attention_out, _ = self.attention(x, mask)
        attention_out = self.dropout_1(attention_out,training=training)
        x = self.layernorm_1(x + attention_out)

        feedForward_out = x
        for layer in self.feedForward:
            feedForward_out = layer(feedForward_out)
        feedForward_out = self.dropout_2(feedForward_out, training=training)
        x = self.layernorm_2(x + feedForward_out)

        return x

class DecoderLayer(Layer):
    def __init__(self, model_dim, num_heads, ffn_units, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()

        self.enc_dec_attention = MultiHeadAttention(model_dim, num_heads)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

        self.feedForward = [
            tf.keras.layers.Dense(ffn_units, activation='relu'),
            tf.keras.layers.Dense(model_dim)
        ]
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
    
    def call(self, x, enc_out, mask, training = False):
        attention_out, _ = self.attention(x, mask)
        attention_out = self.dropout_1(attention_out,training=training)
        x = self.layernorm_1(x + attention_out)

        enc_dec_attention_out, attention_weights = self.enc_dec_attention(x, mask=None, encoder_out=enc_out)
        enc_dec_attention_out = self.dropout_2(enc_dec_attention_out,training=training)
        x = self.layernorm_2(x + enc_dec_attention_out)

        feedForward_out = x
        for layer in self.feedForward:
            feedForward_out = layer(feedForward_out)
        feedForward_out = self.dropout_3(feedForward_out, training=training)
        x = self.layernorm_3(x + feedForward_out)

        return x, attention_weights

class Encoder(Layer):
    def __init__(self, num_layers, model_dim, num_heads, ffn_units, seq_len, dropout_rate):
        super(Encoder, self).__init__()
        self.model_dim = model_dim

        self.embedding = tf.keras.layers.Dense(model_dim)
        self.pos_encoding = utils.positional_encoding(seq_len,model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.enc_layers = [EncoderLayer(model_dim,num_heads,ffn_units,dropout_rate) for _ in range(num_layers)]

    def call(self, x, mask, training=False):
        batch_size = x.shape[0]

        x = tf.reshape(x,(batch_size,-1, x.shape[-1]))

        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, mask, training)

        return x

class Decoder(Layer):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, ffn_units, seq_len, dropout_rate):
        super(Decoder, self).__init__()
        self.model_dim = model_dim

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=model_dim)
        self.pos_encoding = utils.positional_encoding(seq_len, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dec_layers = [DecoderLayer(model_dim,num_heads,ffn_units,dropout_rate) for _ in range(num_layers)]

    def call(self, x, enc_out, mask, training=False):
        seq_len = x.shape[1]
        attention_weights = {}
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len]

        x = self.dropout(x, training=training)

        for i, layer in enumerate(self.dec_layers):
            x, enc_dec_attention = layer(x, enc_out, mask, training)
            
            attention_weights[f'dec_layer_{i}'] = enc_dec_attention

        return x, attention_weights

class Transformer(Model):
    def __init__(self, vocab_size, num_enc_layers, num_dec_layers, model_dim, num_heads, ffn_units, enc_seq_len, dec_seq_len, dropout_rate):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_enc_layers, model_dim, num_heads, ffn_units, enc_seq_len, dropout_rate)
        self.decoder = Decoder(vocab_size, num_dec_layers, model_dim, num_heads, ffn_units, dec_seq_len, dropout_rate)

        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, input, mask=None, training=False):
        x, target = input
        encoder_out = self.encoder(x, mask=None, training=training)
        decoder_out, attention_weights = self.decoder(target, encoder_out, mask, training=training)
        output = self.output_layer(decoder_out)

        return output, attention_weights