import numpy as np
import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):

    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(key.shape[-1],tf.float32))

    # Add the mask to the scaled tensor
    if mask is not None:
        scores += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scores)

    output = attention_weights @ value

    return output, attention_weights

def positional_encoding(seq_len, model_dim):
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
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask