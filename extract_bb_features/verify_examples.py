import tensorflow as tf
import numpy as np
import os
import time

records_path = os.path.join('bb_features', 'flickr8k_test.tfrecords')

for serialized_example in tf.python_io.tf_record_iterator(records_path):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    bounding_box_id = example.features.feature['bounding_box_id'].int64_list.value
    image_id = example.features.feature['image_id'].int64_list.value
    fc7_activations_bytes = example.features.feature['fc7_activations'].bytes_list.value[0]
    fc7_activations = np.fromstring(fc7_activations_bytes, dtype=np.float32)
    print(bounding_box_id, image_id, fc7_activations[:10])
    time.sleep(1)