import preprocessing
import network
import utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)

def plot_result(num_epochs, epoch, train_losses, test_losses):
    fig = plt.figure(figsize=(10,8))
    plt.xlim((0,epoch))
    plt.plot(train_losses,label='Training')
    plt.plot(test_losses,label='Test')
    plt.ylabel('Loss',fontsize=24)
    plt.xlabel('Epochs',fontsize=24)
    plt.title(f'Epoch: {epoch+1}', fontweight='bold', fontsize=30)
    plt.xticks(np.arange(0,epoch,2))    
    plt.legend()
    plt.savefig('train_result.png')
    plt.show()

def train_model(model, train_dataset, val_dataset, optimizer, loss_object, num_epochs, dec_seq_len, tokenizer):
    running_average_factor = 0.95
    
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []

    # Initialize plot
    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xlim((0,num_epochs))
    train_loss_plot, = ax1.plot(train_losses,label='Training')
    test_loss_plot, = ax1.plot(test_losses,label='Training')
    ax1.set_ylabel('Loss',fontsize=24)
    ax1.set_xlabel('Epochs',fontsize=24)
    ax1.set_title(f'Epoch: 0', fontweight='bold', fontsize=30)
    ax1.set_xticks(np.arange(0,num_epochs,2))    
    ax1.legend()

    # create mask
    mask = utils.look_ahead_mask(size = dec_seq_len)
    
    # Train loop for num_epochs epochs.
    for epoch in range(num_epochs):
            
        # Training
        running_average_loss = 0
        for input_seq, target_seq in train_dataset:
            train_loss = train_step(model, input_seq, target_seq, mask, loss_object, optimizer)
            running_average_loss = running_average_factor * running_average_loss  + (1 - running_average_factor) * train_loss

        train_losses.append(running_average_loss.numpy())

        # Test
        total_test_loss = 0
        for input_seq, target_seq in val_dataset:
            test_loss = test_step(model, input_seq, target_seq, loss_object, tokenizer)
            total_test_loss += test_loss

        test_losses.append(total_test_loss)


        # Display loss and accuracy for current epoch    
        print(f'Epoch: __ {epoch+1}')
        print('Train loss: ',running_average_loss.numpy())

        # Update plot
        train_loss_plot.set_xdata(np.arange(epoch+1))
        test_loss_plot.set_xdata(np.arange(epoch+1))

        train_loss_plot.set_ydata(train_losses)
        test_loss_plot.set_ydata(test_losses)
        ax1.set_ylim((0,max(train_losses)))
        ax1.set_title(f'Epoch: {epoch+1}', fontweight='bold', fontsize=30)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    plt.ioff()
    plt.close()
    plot_result(num_epochs, epoch, train_losses, test_losses)

@tf.function
def train_step(model, input_seq, target_seq, look_ahead_mask, loss_object, optimizer):
    # target_seq_prior is the input for the decoder utilizing teacher-forcing, it does not contain the <end> token
    target_seq_prior = target_seq[:, :-1]
    # target_seq_posterior is the target output, it does not contain the <start> token
    target_seq_posterior = target_seq[:, 1:]

    with tf.GradientTape() as tape:
        pred, _ = model((input_seq, target_seq_prior), look_ahead_mask, training=True)
        loss = loss_function(target_seq_posterior, pred, loss_object)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(model, input_seq, target_seq, loss_object, tokenizer):
    batch_size = target_seq.shape[0]
    output = tf.expand_dims(tf.repeat([tokenizer.word_index['<start>']],batch_size), 1)
    for _ in range(dec_seq_len):
        mask = utils.look_ahead_mask(output.shape[1])
        pred, _ = model((input_seq, output), mask)

        # Take the last decoder output
        pred = pred[:, -1:, :]

        # Get the predicted word
        predicted_id = tf.argmax(pred, axis=-1, output_type=tf.int32)

        try:
            softmax_pred = tf.concat([softmax_pred,pred], axis=1)
        except:
            softmax_pred = pred
                
        # Add latest prediction to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)
    
    loss = loss_function(target_seq[:, 1:], softmax_pred, loss_object)

    return loss

def loss_function(target, pred, loss_object):
    # Create mask to ignore padding
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss = loss_object(target, pred)

    mask = tf.cast(mask, dtype=loss.dtype)

    # Set loss for padded values to 0
    loss *= mask

    # Calculate mean loss
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss

if __name__ == "__main__":

    num_patches_h = 4
    num_patches_v = 4
    vocab_size = 5000

    train_dataset, val_dataset, max_seq_len, tokenizer = preprocessing.get_datasets(image_dim=(64, 64), num_patches_h = num_patches_h, num_patches_v = num_patches_v, vocab_size = vocab_size)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    tf.keras.backend.clear_session()

    num_enc_layers = 6
    num_dec_layers = 2
    model_dim = 32
    num_heads = 4
    ffn_units = 64
    enc_seq_len = num_patches_h*num_patches_v
    dec_seq_len = max_seq_len - 1
    dropout_rate = 0.3
    epochs = 20

    learning_rate = CustomSchedule(model_dim)

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    # Initialize model
    captioning_model = network.Transformer(vocab_size, num_enc_layers, num_dec_layers, model_dim, num_heads, ffn_units, enc_seq_len, dec_seq_len, dropout_rate)

    train_model(
        model = captioning_model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        optimizer = optimizer,
        loss_object = loss_object,
        num_epochs = epochs,
        dec_seq_len = dec_seq_len,
        tokenizer = tokenizer
    )

    captioning_model.save_weights('captioning_model.h5')