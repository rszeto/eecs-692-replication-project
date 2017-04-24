import numpy as np
import tensorflow as tf
import os
import sys
import time
from eval_utils import S_kl_np

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

S_kl_compute_batch_size = 400

def main():
    dataset = 'flickr8k'
    train_split = 'train'
    train_use_one_caption_per_image = False
    training_run_name = 'Sun_Apr_23_18:37:19_2017'
    snapshot_iter_num = 10000
    eval_split = 'dev'
    eval_use_one_caption_per_image = False
    eval_max_num_pairs = -1  # Set to -1 to ignore
    store_S_kl = True

    if train_use_one_caption_per_image:
        train_split += '_one_caption'
    train_split_key = '%s_%s' % (dataset, train_split)
    training_run_root = os.path.join(SCRIPT_DIR, 'training_runs', train_split_key, training_run_name)

    # Import version of model defined in the training run
    sys.path.insert(0, training_run_root)
    import model_utils
    from model_utils import AlignmentModel
    from model_utils import RegionCaptionDataLoader

    ### Get dataset information ###
    if dataset == 'flickr8k':
        dataset_wrapper = Flickr8kFactory().get_dataset_wrapper()
    elif dataset == 'mscoco':
        dataset_wrapper = MSCOCOFactory().get_dataset_wrapper()
    else:
        print('Unknown dataset %s' % dataset)
        return

    dictionary = dataset_wrapper.get_dictionary()
    max_caption_length = dataset_wrapper.get_max_caption_length()
    get_eval_data_fn_name = 'get_%s_data' % eval_split
    eval_data = getattr(dataset_wrapper, get_eval_data_fn_name)()
    num_pairs = len(eval_data['caption_data']) if eval_max_num_pairs <= 0 else eval_max_num_pairs

    # Create map from caption ID to caption length
    caption_id_to_caption_length_map = {datum['caption_id']: len(datum['caption']) for datum in eval_data['caption_data']}

    if eval_use_one_caption_per_image:
        eval_split += '_one_caption'
    if eval_max_num_pairs > 0:
        eval_split += '_%d_pairs' % eval_max_num_pairs
    eval_record_path = os.path.join(SCRIPT_DIR, 'paired_data', '%s_%s.tfrecords' % (dataset, eval_split))
    data_loader = RegionCaptionDataLoader(eval_record_path, len(dictionary), max_caption_length, shuffle=False)
    batch_image_ids = data_loader.get_batch_image_ids()
    batch_caption_ids = data_loader.get_batch_caption_ids()
    batch_fc7_activation_sets = data_loader.get_batch_fc7_activation_sets()
    batch_image_captions = data_loader.get_batch_image_captions()
    batch_bounding_box_ids = data_loader.get_batch_bounding_box_ids()

    model = AlignmentModel(batch_fc7_activation_sets, batch_image_captions, len(dictionary), max_caption_length, test_only=True)
    region_embeddings = model.get_region_embeddings()
    word_embeddings = model.get_word_embeddings()

    # Saver to load checkpoint
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(training_run_root, 'checkpoint-%d' % snapshot_iter_num)

    # Store image/caption IDS and embeddings in matrices
    all_image_ids = np.zeros(num_pairs)
    all_caption_ids = np.zeros(num_pairs)
    all_region_embeddings = np.zeros((num_pairs, model_utils.joint_embedding_size, 20))
    all_word_embeddings = np.zeros((num_pairs, model_utils.joint_embedding_size, max_caption_length))
    all_bounding_box_ids = np.zeros((num_pairs, model_utils.NUM_IMAGE_REGIONS))

    batch_size = model_utils.batch_size
    joint_embedding_size = model_utils.joint_embedding_size

    with tf.Session() as sess:
        # Start implicit queue from read_and_decode
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Load checkpoint
        saver.restore(sess, checkpoint_path)

        num_batches = int(np.ceil(float(num_pairs) / batch_size))
        final_batch_size = num_pairs - (num_batches - 1) * batch_size
        start_time = time.time()
        print('Starting forward pass')

        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else final_batch_size
            cur_image_ids, cur_caption_ids, cur_bounding_box_ids, cur_region_embeddings, cur_word_embeddings = model.run_inference(sess, [batch_image_ids, batch_caption_ids, batch_bounding_box_ids, region_embeddings, word_embeddings])
            all_image_ids[batch_size*i:batch_size*i+cur_batch_size] = cur_image_ids
            all_caption_ids[batch_size*i:batch_size*i+cur_batch_size] = cur_caption_ids
            all_region_embeddings[batch_size*i:batch_size*i+cur_batch_size, :, :] = cur_region_embeddings
            all_word_embeddings[batch_size*i:batch_size*i+cur_batch_size, :, :] = cur_word_embeddings
            all_bounding_box_ids[batch_size*i:batch_size*i+cur_batch_size, :] = cur_bounding_box_ids

        end_time = time.time()
        print('Forward pass duration: %f' % (end_time - start_time))

        # Stop threads
        coord.request_stop()
        coord.join(threads)

    # Filter out repeated images
    print('Filtering out repeated images')
    unique_image_ids, unique_image_id_indexes = np.unique(all_image_ids, return_index=True)
    all_region_embeddings = all_region_embeddings[unique_image_id_indexes, :, :]
    all_bounding_box_ids = all_bounding_box_ids[unique_image_id_indexes, :]

    # Make save directory
    save_path = os.path.join(SCRIPT_DIR, 'eval_results', '%s_%s_%s_%s' % (dataset, eval_split, training_run_name, snapshot_iter_num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save embeddings
    print('Saving embeddings at %s' % save_path)
    np.save(os.path.join(save_path, 'image_ids'), unique_image_ids)
    np.save(os.path.join(save_path, 'caption_ids'), all_caption_ids)
    np.save(os.path.join(save_path, 'bounding_box_ids'), all_bounding_box_ids)
    np.save(os.path.join(save_path, 'region_embeddings'), all_region_embeddings)
    np.save(os.path.join(save_path, 'word_embeddings'), all_word_embeddings)

    # Construct S_kl
    if store_S_kl:
        print('Constructing S_kl')
        start_time = time.time()
        S_kl = np.zeros((all_region_embeddings.shape[0], all_word_embeddings.shape[0]))
        for i in range(num_pairs / S_kl_compute_batch_size):
            caption_lengths = [caption_id_to_caption_length_map[caption_id] for caption_id in all_caption_ids[S_kl_compute_batch_size*i:S_kl_compute_batch_size*(i+1)]]
            S_kl[:, S_kl_compute_batch_size*i:S_kl_compute_batch_size*(i+1)] = S_kl_np(all_region_embeddings, all_word_embeddings[S_kl_compute_batch_size*i:S_kl_compute_batch_size*(i+1), :, :], caption_lengths)
            print('Processed %d/%d columns' % (S_kl_compute_batch_size*(i+1), num_pairs))

        end_time = time.time()
        print('S_kl computation: %f\n' % (end_time - start_time))  # Takes ~2 mins for 1k images and 5k captions
        print('Saving S_kl')
        np.save(os.path.join(save_path, 'S_kl'), S_kl)

    print('Saved all data')

if __name__ == '__main__':
    main()