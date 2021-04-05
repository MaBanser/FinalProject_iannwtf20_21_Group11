import numpy as np
import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):
    # Calculate the attention scores from query and key
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(key.shape[-1],tf.float32))

    # Add a negative mask so that the softmax-scores for values that the model shouldn't attend to will be near 0
    if mask is not None:
        scores += (mask * -1e9)  

    # Calculate the softmax score
    attention_weights = tf.nn.softmax(scores)

    # Weight values by softmax score
    output = attention_weights @ value

    return output, attention_weights

def positional_encoding(seq_len, model_dim):
    # Calculate the positional encoding following the formulation of the 'Attention is all you need' paper
    pos = np.expand_dims(np.arange(seq_len),1)
    i = np.expand_dims(np.arange(model_dim),0)
    angle_rads = pos/(10000**(2*(i//2)/model_dim))

    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Add batch dimension
    pos_encoding = np.expand_dims(angle_rads,0)

    return tf.cast(pos_encoding, dtype=tf.float32)

def look_ahead_mask(size):
    # Create a triangular matrix filled with ones to prevent the decoder from attending to it's future prediction targets
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask