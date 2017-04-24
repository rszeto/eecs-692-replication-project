import numpy as np
import tensorflow as tf
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
# Import dataset wrappers
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

DATASET = 'flickr8k'
DATA_SPLIT = 'train'
USE_ONE_CAPTION_PER_IMAGE = True
MAX_NUM_PAIRS = 1000  # Set to -1 to use all pairs

### Utility functions ###
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
### End utility function ###

### Get dataset information ###
if DATASET == 'flickr8k':
    dataset_wrapper = Flickr8kFactory().get_dataset_wrapper()
elif DATASET == 'mscoco':
    dataset_wrapper = MSCOCOFactory().get_dataset_wrapper()
else:
    print('Unknown dataset %s' % DATASET)
    exit()

if DATA_SPLIT == 'train':
    split_data = dataset_wrapper.get_train_data()
elif DATA_SPLIT == 'dev':
    split_data = dataset_wrapper.get_dev_data()
elif DATA_SPLIT == 'test':
    split_data = dataset_wrapper.get_test_data()
else:
    print('Unknown data split %s' % DATA_SPLIT)
    exit()
### End get dataset information ###

dictionary = dataset_wrapper.get_dictionary()
caption_data = split_data['caption_data']
max_caption_length = dataset_wrapper.get_max_caption_length()

# Create index from image_id to captions
caption_rev_ind = {}
for datum in caption_data:
    # Convert caption into array of one-hot encodings
    caption_id = datum['caption_id']
    caption_as_word_indexes = datum['caption']
    caption_as_one_hot_vecs = np.zeros((len(dictionary), max_caption_length))
    for i in range(len(caption_as_word_indexes)):
        caption_as_one_hot_vecs[caption_as_word_indexes[i], i] = 1
    image_id = datum['image_id']
    if image_id not in caption_rev_ind:
        caption_rev_ind[image_id] = []
    caption_rev_ind[image_id].append((caption_id, caption_as_one_hot_vecs))

# Define paths to fc7 activations and paired data
fc7_records_path = os.path.join(SCRIPT_DIR, 'extract_bb_features', 'bb_features', '%s_%s.tfrecords' % (DATASET, DATA_SPLIT))
if USE_ONE_CAPTION_PER_IMAGE:
    DATA_SPLIT += '_one_caption'
if MAX_NUM_PAIRS > 0:
    DATA_SPLIT += '_%d_pairs' % MAX_NUM_PAIRS
paired_data_records_path = os.path.join(SCRIPT_DIR, 'paired_data', '%s_%s.tfrecords'
                                        % (DATASET, DATA_SPLIT))

if not os.path.exists(os.path.dirname(paired_data_records_path)):
    os.makedirs(os.path.dirname(paired_data_records_path))
if os.path.exists(paired_data_records_path):
    os.remove(paired_data_records_path)
writer = tf.python_io.TFRecordWriter(paired_data_records_path)

feature_store = {}
bounding_box_id_store = {}
num_written_examples = 0
for serialized_example in tf.python_io.tf_record_iterator(fc7_records_path):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    bounding_box_id = example.features.feature['bounding_box_id'].int64_list.value[0]
    image_id = example.features.feature['image_id'].int64_list.value[0]
    fc7_activations_bytes = example.features.feature['fc7_activations'].bytes_list.value[0]
    fc7_activations = np.fromstring(fc7_activations_bytes, dtype=np.float32)

    if image_id not in feature_store:
        feature_store[image_id] = []
        bounding_box_id_store[image_id] = []
    feature_store[image_id].append(fc7_activations)
    bounding_box_id_store[image_id].append(bounding_box_id)

    # If all bounding boxes have been found, prepare data
    if len(bounding_box_id_store[image_id]) == 20:
        if image_id % 100 == 0:
            print('Found all boxes for image %d, saving paired data' % image_id)
        bounding_box_ids_bytes = np.array(bounding_box_id_store[image_id], dtype=np.int32).tobytes()
        fc7_activation_set_bytes = np.array(feature_store[image_id], dtype=np.float32).tobytes()
        image_captions = caption_rev_ind[image_id]
        for caption_id, image_caption in image_captions:
            image_caption_bytes = image_caption.astype(np.int32).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_id': _int64_feature(image_id),
                'caption_id': _int64_feature(caption_id),
                'bounding_box_ids': _bytes_feature(bounding_box_ids_bytes),
                'fc7_activation_set': _bytes_feature(fc7_activation_set_bytes),
                'image_caption': _bytes_feature(image_caption_bytes)
            }))
            writer.write(example.SerializeToString())
            num_written_examples += 1
            if USE_ONE_CAPTION_PER_IMAGE:
                break

        # Clear elements in stores from memory
        del feature_store[image_id]
        del bounding_box_id_store[image_id]

        if MAX_NUM_PAIRS > 0 and num_written_examples >= MAX_NUM_PAIRS:
            break

# Stop writing
writer.close()
print('Max caption length: %d' % max_caption_length)
print('Num written examples: %d' % num_written_examples)