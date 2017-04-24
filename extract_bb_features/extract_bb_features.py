import numpy as np
import tensorflow as tf
import subprocess
import os
import sys
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imsave
import pdb

SCRIPT_DIR = os.path.dirname(__file__)
CAFFE_TF_DIR = os.path.join(SCRIPT_DIR, 'caffe-tensorflow')
BATCH_SIZE = 500
NUM_CONCURRENT = 4
IMAGENET_MEAN = [104, 117, 124]
DATASET = 'flickr8k'
DATA_SPLIT = 'train'


### Utility functions ###
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
### End utility function ###

# Get ImageNet class names
with open(os.path.join(CAFFE_TF_DIR, 'examples', 'imagenet', 'imagenet-classes.txt'), 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Import dataset wrappers
sys.path.append(os.path.join('..', '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join('..', '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

# Create TensorFlow versions of AlexNet code and weights if not existent
if not os.path.exists('alexnet_weights.npy') \
        or not os.path.exists('alexnet_code.py'):
    conversion_res = subprocess.call([
        'python', os.path.join(CAFFE_TF_DIR, 'convert.py'),
        'alexnet_deploy.prototxt',
        '--code-output-path', 'alexnet_code.py',
        '--caffemodel', 'bvlc_alexnet.caffemodel',
        '--data-output-path', 'alexnet_weights.npy'
    ])
    assert(conversion_res == 0)

# Import AlexNet from generated code
sys.path.append(CAFFE_TF_DIR)
from alexnet_code import AlexNet

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

image_data = split_data['image_data']
caption_data = split_data['caption_data']
bounding_box_data = split_data['bounding_box_data']
dictionary = dataset_wrapper.get_dictionary()
num_regions = len(bounding_box_data)

# Convert image IDs in bounding box data to files
image_id_path_map = {datum['image_id']: datum['file_path'] for datum in image_data}
bounding_boxes_list = np.array([datum['bounding_box'] for datum in bounding_box_data])
bounding_box_ids_list = np.array([datum['bounding_box_id'] for datum in bounding_box_data])
file_paths_list = [image_id_path_map[datum['image_id']] for datum in bounding_box_data]
image_id_list = [datum['image_id'] for datum in bounding_box_data]

# Queue for bounding box info (ID, file path, and bounding box)
bounding_box_data_queue = tf.FIFOQueue(capacity=num_regions, dtypes=[tf.int32, tf.int32, tf.string, tf.int32])
enqueue_bounding_box_data_op = bounding_box_data_queue.enqueue_many([bounding_box_ids_list, image_id_list, file_paths_list, bounding_boxes_list])
bounding_box_id, image_id, file_path, bounding_box = bounding_box_data_queue.dequeue()

# Operations to convert bounding box info to actual region
file_data = tf.read_file(file_path)
image = tf.image.decode_jpeg(file_data, channels=3)
image = tf.reverse(image, [2])
crop = tf.image.crop_to_bounding_box(image, bounding_box[1], bounding_box[0], bounding_box[3] - bounding_box[1], bounding_box[2] - bounding_box[0])
crop = tf.expand_dims(crop, 0)
resized_crop = tf.image.resize_images(crop, [227, 227])
resized_crop_mean_sub = resized_crop - IMAGENET_MEAN  # Mean subtraction
resized_crop_mean_sub = tf.squeeze(resized_crop_mean_sub)  # Remove batch dimension
resized_crop_rgb = tf.cast(tf.squeeze(tf.reverse(resized_crop, [3])), tf.uint8)

# Queue for actual regions (ID, image ID, and cropped image)
region_queue = tf.FIFOQueue(capacity=BATCH_SIZE * NUM_CONCURRENT,
                            dtypes=[tf.int32, tf.int32, tf.float32, tf.uint8],
                            shapes=[(), (), (227, 227, 3), (227, 227, 3)])
enqueue_region_op = region_queue.enqueue([bounding_box_id, image_id, resized_crop_mean_sub, resized_crop_rgb])
bounding_box_ids, image_ids, region, region_vis = region_queue.dequeue_many(BATCH_SIZE)

# Initialize AlexNet
net = AlexNet({'data': region})
fc7_activations = net.get_activations('fc7')

# Queue for fc7 activations
fc7_queue = tf.FIFOQueue(capacity=BATCH_SIZE * NUM_CONCURRENT,
                         dtypes=[tf.int32, tf.int32, tf.float32],
                         shapes=[(), (), (4096)])
enqueue_fc7_op = fc7_queue.enqueue_many([bounding_box_ids, image_ids, fc7_activations])
bounding_box_id, image_id, fc7_activations = fc7_queue.dequeue()

# Initialize QueueRunner
qr = tf.train.QueueRunner(bounding_box_data_queue, [enqueue_bounding_box_data_op])
tf.train.add_queue_runner(qr)
qr = tf.train.QueueRunner(region_queue, [enqueue_region_op])
tf.train.add_queue_runner(qr)
qr = tf.train.QueueRunner(fc7_queue, [enqueue_fc7_op] * NUM_CONCURRENT)
tf.train.add_queue_runner(qr)

# Keep track of written bounding box IDs
written_bounding_box_ids = np.zeros(num_regions)

with tf.Session() as sess:
    # Initialize AlexNet weights
    net.load('alexnet_weights.npy', sess)

    # Start queue runner, which will populate data asynchronously
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Define records file path
    records_path = os.path.join('bb_features', '%s_%s.tfrecords' % (DATASET, DATA_SPLIT))
    # Make bb_features folder
    if not os.path.exists('bb_features'):
        os.makedirs('bb_features')
    # Remove old record file if it exists
    if os.path.exists(records_path):
        os.remove(records_path)

    # Start writing
    writer = tf.python_io.TFRecordWriter(records_path)
    for i in range(num_regions):
        if i % 1000 == 0:
            print('Processed %d/%d regions' % (i, num_regions))
        my_bounding_box_id, my_image_id, my_fc7_activations = sess.run([bounding_box_id, image_id, fc7_activations])
        my_fc7_activations_bytes = my_fc7_activations.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'bounding_box_id': _int64_feature(my_bounding_box_id),
            'image_id': _int64_feature(my_image_id),
            'fc7_activations': _bytes_feature(my_fc7_activations_bytes)
        }))
        writer.write(example.SerializeToString())
        # Store bounding box ID
        written_bounding_box_ids[i] = my_bounding_box_id

    # Stop writing
    writer.close()

    # Stop the threads and wait for them to finish
    coord.request_stop()
    coord.join(threads)

print('Wrote %d unique bounding boxes' % len(np.unique(written_bounding_box_ids)))