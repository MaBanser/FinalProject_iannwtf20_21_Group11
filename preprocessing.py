import os
import json

import tensorflow as tf

def download_data():
    # Download train image files
    train_image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + train_image_folder):
        train_image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                        extract = True)
        train_image_folder = os.path.dirname(train_image_zip) + train_image_folder
        os.remove(train_image_zip)
    else:
        train_image_folder = os.path.abspath('.') + train_image_folder


    # Download validation image files
    val_image_folder = '/val2014/'
    if not os.path.exists(os.path.abspath('.') + val_image_folder):
        val_image_zip = tf.keras.utils.get_file('val2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/val2014.zip',
                                        extract = True)
        val_image_folder = os.path.dirname(val_image_zip) + val_image_folder
        os.remove(val_image_zip)
    else:
        val_image_folder = os.path.abspath('.') + val_image_folder


    # Download caption annotation files for training and validation
    annotation_folder = '/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                            extract = True)
        train_annotations_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
        val_annotations_file = os.path.dirname(annotation_zip)+'/annotations/captions_val2014.json'
        os.remove(annotation_zip)
    else:
        train_annotations_file = os.path.abspath('.')+'/annotations/captions_train2014.json'
        val_annotations_file = os.path.abspath('.')+'/annotations/captions_val2014.json'

    return train_image_folder, val_image_folder, train_annotations_file, val_annotations_file

def load_json(json_file):
    with open(json_file, 'r') as file:
        annotations = json.load(file)
    return annotations

# Group all train and validation captions together having the same image ID.
def sort_captions(train_annotations, val_annotations):
    train_image_id_to_captions = {}
    train_image_ids = []
    for anot in train_annotations['annotations']:
        caption = f"<start> {anot['caption']} <end>"
        train_image_id = 'COCO_train2014_' + '%012d' % (anot['image_id'])
        try:
            train_image_id_to_captions[train_image_id].append(caption)
        except:
            train_image_ids.append(train_image_id)
            train_image_id_to_captions[train_image_id] = [caption]

    val_image_id_to_captions = {}
    val_image_ids = []
    for anot in val_annotations['annotations']:
        caption = f"<start> {anot['caption']} <end>"
        val_image_id = 'COCO_val2014_' + '%012d' % (anot['image_id'])
        try:
            val_image_id_to_captions[val_image_id].append(caption)
        except:
            val_image_ids.append(val_image_id)
            val_image_id_to_captions[val_image_id] = [caption]
    
    return train_image_id_to_captions, train_image_ids, val_image_id_to_captions, val_image_ids

# Loading and resizing image
def load_image(image_path, image_dim):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_dim, method='nearest')
    return img

# Divide image in equal patches
def create_patches(image, patch_height, patch_width, batch = True):
    if not batch:
        image = tf.expand_dims(image, 0)
    patches = tf.image.extract_patches(images=image,
                           sizes=[1, patch_height, patch_width, 1],
                           strides=[1, patch_height, patch_width, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    if not batch:
        patches = tf.squeeze(patches)
    return patches

def get_captions(train_image_ids, val_image_ids, train_image_id_to_captions, val_image_id_to_captions, train_image_folder, val_image_folder):
    train_captions = []
    val_captions = []

    train_image_path = []
    val_image_path = []

    for img_id in train_image_ids:
        caption_list = train_image_id_to_captions[img_id]
        train_captions.extend(caption_list)
        train_image_path.extend([train_image_folder + img_id + '.jpg'] * len(caption_list))

    for img_id in val_image_ids:
        caption_list = val_image_id_to_captions[img_id]
        val_captions.extend(caption_list)
        val_image_path.extend([val_image_folder + img_id + '.jpg'] * len(caption_list))

    return train_captions, val_captions, train_image_path, val_image_path

def tokenize_captions(train_captions, val_captions, vocab_size):
    all_captions = train_captions + val_captions

    # Choose the top vocab_size words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                      oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(all_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors for train captions
    train_captions_tokenized = tokenizer.texts_to_sequences(train_captions)
    val_captions_tokenized = tokenizer.texts_to_sequences(val_captions)
        
    max_seq_len = max(len(seq) for seq in train_captions_tokenized + val_captions_tokenized)

    # Pad sequences
    train_captions_padded = tf.keras.preprocessing.sequence.pad_sequences(train_captions_tokenized, maxlen=max_seq_len, padding='post')
    val_captions_padded = tf.keras.preprocessing.sequence.pad_sequences(val_captions_tokenized, maxlen=max_seq_len, padding='post')

    return max_seq_len, train_captions_padded, val_captions_padded, tokenizer

def get_datasets(batch_size = 8, image_dim = (200, 200), num_patches_h = 8, num_patches_v = 8, vocab_size=5000):
    patch_height = image_dim[0]//num_patches_v
    patch_width = image_dim[1]//num_patches_h

    train_image_folder, val_image_folder, train_annotations_file, val_annotations_file = download_data()

    train_annotations = load_json(train_annotations_file)
    val_annotations = load_json(val_annotations_file)

    train_image_id_to_captions, train_image_ids, val_image_id_to_captions, val_image_ids = sort_captions(train_annotations, val_annotations)

    train_captions, val_captions, train_image_path, val_image_path = get_captions(train_image_ids, 
                                                                                  val_image_ids, 
                                                                                  train_image_id_to_captions, 
                                                                                  val_image_id_to_captions, 
                                                                                  train_image_folder, 
                                                                                  val_image_folder)

    max_seq_len, train_captions_padded, val_captions_padded, tokenizer = tokenize_captions(train_captions, val_captions, vocab_size)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_path, train_captions_padded))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_path, val_captions_padded))

    # Shuffle dataset
    train_dataset = train_dataset.shuffle(batch_size)
    val_dataset = val_dataset.shuffle(batch_size)

    # Map paths to images and create batches
    train_dataset = train_dataset.map(lambda img_path, cap: (load_image(img_path, image_dim), cap),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    val_dataset = val_dataset.map(lambda img_path, cap: (load_image(img_path, image_dim), cap),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

    # # Map images to patches
    train_dataset = train_dataset.map(lambda img, cap: (create_patches(img,patch_height,patch_width), cap),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(lambda img, cap: (create_patches(img,patch_height,patch_width), cap),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, max_seq_len, tokenizer
