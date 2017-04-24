import numpy as np
import tensorflow as tf
import os
import sys
import time
import glob
import model_utils
from model_utils import AlignmentModel
from model_utils import RegionCaptionDataLoader
import pdb

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory


def main():
    dataset = 'flickr8k'
    training_run_name = 'Sun_Apr_23_18:39:53_2017'
    snapshot_iter_num = 10000
    eval_split = 'dev'
    eval_use_one_caption_per_image = False
    eval_max_num_pairs = -1  # Set to -1 to ignore

    ### Get dataset information ###
    if dataset == 'flickr8k':
        dataset_wrapper = Flickr8kFactory().get_dataset_wrapper()
    elif dataset == 'mscoco':
        dataset_wrapper = MSCOCOFactory().get_dataset_wrapper()
    else:
        print('Unknown dataset %s' % dataset)
        return

    get_eval_data_fn_name = 'get_%s_data' % eval_split
    eval_data = getattr(dataset_wrapper, get_eval_data_fn_name)()

    if eval_use_one_caption_per_image:
        eval_split += '_one_caption'
    if eval_max_num_pairs > 0:
        eval_split += '_%d_pairs' % eval_max_num_pairs

    print('Now evaluating model %s, iter %d on %s split' % (training_run_name, snapshot_iter_num, eval_split))

    # Create index from image ID to list of GT caption IDs
    image_to_caption_ids_map = {}
    caption_to_image_ids_map = {}
    for datum in eval_data['caption_data']:
        caption_id = datum['caption_id']
        image_id = datum['image_id']
        if image_id not in image_to_caption_ids_map:
            image_to_caption_ids_map[image_id] = []
        image_to_caption_ids_map[image_id].append(caption_id)
        caption_to_image_ids_map[caption_id] = image_id

    # Load image and caption IDs and embeddings
    print('Loading saved IDs and embeddings')
    results_path = os.path.join(SCRIPT_DIR, 'eval_results', '%s_%s_%s_%s' % (dataset, eval_split, training_run_name, snapshot_iter_num))
    all_image_ids = np.load(os.path.join(results_path, 'image_ids.npy'))
    all_caption_ids = np.load(os.path.join(results_path, 'caption_ids.npy'))
    S_kl = np.load(os.path.join(results_path, 'S_kl.npy'))

    # Open results file
    results_file = open(os.path.join(results_path, 'results.txt'), 'w')

    ### Image ranking evaluation ###
    ranked_image_indexes = np.argsort(S_kl, axis=0)
    ranked_image_ids = -1 * np.ones(S_kl.shape)
    for i in range(S_kl.shape[0]):
        ranked_image_ids[ranked_image_indexes == i] = all_image_ids[i]
    # Flip rows so most confident images are first
    ranked_image_ids[...] = ranked_image_ids[::-1, :]
    # For each caption, find where the GT image is listed
    final_image_ranks = np.zeros(len(all_caption_ids))
    for j, caption_id in enumerate(all_caption_ids):
        final_image_ranks[j] = np.where(ranked_image_ids[:, j] == caption_to_image_ids_map[caption_id])[0]

    # Print metrics
    recall_at_1 = np.sum(final_image_ranks < 1) / float(len(all_caption_ids))
    recall_at_5 = np.sum(final_image_ranks < 5) / float(len(all_caption_ids))
    recall_at_10 = np.sum(final_image_ranks < 10) / float(len(all_caption_ids))
    med_r = np.median(final_image_ranks) + 1
    results_file.write('Image retrieval:\t')
    results_file.write('R@1 = %.3f\t\t' % recall_at_1)
    results_file.write('R@5 = %.3f\t\t' % recall_at_5)
    results_file.write('R@10 = %.3f\t\t' % recall_at_10)
    results_file.write('Medr = %d\n' % med_r)

    ranked_caption_indexes = np.argsort(S_kl, axis=1)
    ranked_caption_ids = -1 * np.ones(S_kl.shape)
    for i in range(S_kl.shape[1]):
        ranked_caption_ids[ranked_caption_indexes == i] = all_caption_ids[i]
    # Flip columns so most confident captions are first
    ranked_caption_ids[...] = ranked_caption_ids[:, ::-1]

    # In each row, find where best-ranked GT captions occur
    final_caption_ranks = np.zeros(len(all_image_ids))
    for j, image_id in enumerate(all_image_ids):
        # Get ranks of GT captions for current image
        gt_ranks = np.where(np.in1d(ranked_caption_ids[j, :], image_to_caption_ids_map[image_id]))[0]
        # Store minimum rank
        final_caption_ranks[j] = np.min(gt_ranks)

    # Print metrics
    recall_at_1 = np.sum(final_caption_ranks < 1) / float(len(all_image_ids))
    recall_at_5 = np.sum(final_caption_ranks < 5) / float(len(all_image_ids))
    recall_at_10 = np.sum(final_caption_ranks < 10) / float(len(all_image_ids))
    med_r = np.median(final_caption_ranks) + 1
    results_file.write('Sentence retrieval:\t')
    results_file.write('R@1 = %.3f\t\t' % recall_at_1)
    results_file.write('R@5 = %.3f\t\t' % recall_at_5)
    results_file.write('R@10 = %.3f\t\t' % recall_at_10)
    results_file.write('Medr = %d\n' % med_r)

    results_file.close()

if __name__ == '__main__':
    main()