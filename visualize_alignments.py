import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import model_utils
from scipy.misc import imread

HEX_COLOR_LIST = [
    '#00FFFF',  # Cyan
    '#7FFFD4',  # Aquamarine
    '#0000FF',  # Blue
    '#8A2BE2',  # Blue violet
    '#7FFF00',  # Chartreuse (green)
    '#00FF00',  # Green
    '#DC143C',  # Crimson
    '#FF0000',  # Red
    '#FF8C00',  # Dark orange
    '#E9967A',  # Dark salmon
    '#00CED1',  # Dark turquoise
    '#FF1493',  # Deep pink
    '#00BFFF',  # Deep sky blue
    '#FF00FF',  # Magenta
    '#FFFF00'   # Yellow
]

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

def main():
    dataset = 'flickr8k'
    eval_split = 'dev'
    eval_use_one_caption_per_image = False
    eval_max_num_pairs = -1  # Set to -1 to ignore
    training_run_name = 'Sun_Apr_23_21:38:18_2017'
    snapshot_iter_num = 124000
    good_alignment_threshold = 0.25

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
    dictionary = dataset_wrapper.get_dictionary()

    if eval_use_one_caption_per_image:
        eval_split += '_one_caption'
    if eval_max_num_pairs > 0:
        eval_split += '_%d_pairs' % eval_max_num_pairs

    # Load image and caption IDs and embeddings
    print('Loading saved IDs and embeddings')
    results_path = os.path.join(SCRIPT_DIR, 'eval_results', '%s_%s_%s_%s' % (dataset, eval_split, training_run_name, snapshot_iter_num))
    all_image_ids = np.load(os.path.join(results_path, 'image_ids.npy'))
    all_caption_ids = np.load(os.path.join(results_path, 'caption_ids.npy'))
    S_kl = np.load(os.path.join(results_path, 'S_kl.npy'))
    all_region_embeddings = np.load(os.path.join(results_path, 'region_embeddings.npy'))
    all_word_embeddings = np.load(os.path.join(results_path, 'word_embeddings.npy'))
    all_bounding_box_ids = np.load(os.path.join(results_path, 'bounding_box_ids.npy'))

    best_caption_indexes = np.argmax(S_kl, axis=1)
    best_caption_ids = [all_caption_ids[best_caption_index] for best_caption_index in best_caption_indexes]

    image_id_to_file_path_map = {datum['image_id']: datum['file_path'] for datum in eval_data['image_data']}
    caption_id_to_caption_map = {datum['caption_id']: datum['caption'] for datum in eval_data['caption_data']}
    bb_id_to_bb_map = {datum['bounding_box_id']: datum['bounding_box'] for datum in eval_data['bounding_box_data']}

    fig = plt.figure()

    for i, image_id in enumerate(all_image_ids):
        best_caption_index = best_caption_indexes[i]
        best_caption_id = best_caption_ids[i]
        bounding_box_ids = all_bounding_box_ids[i, :]
        image_file_path = image_id_to_file_path_map[image_id]
        caption = caption_id_to_caption_map[best_caption_id]

        region_embeddings = all_region_embeddings[i, :, :]
        word_embeddings = all_word_embeddings[best_caption_index, :, :]
        print(np.linalg.norm(region_embeddings, axis=0))
        print(np.linalg.norm(word_embeddings, axis=0))
        region_word_sim_mat = np.dot(region_embeddings.T, word_embeddings)
        # Find best region for each word
        best_region_indexes = np.argmax(region_word_sim_mat, axis=0)
        best_region_ids = bounding_box_ids[best_region_indexes]
        alignment_scores = np.max(region_word_sim_mat, axis=0)

        image = imread(image_file_path)
        ax = fig.add_axes([.05, .05, .7, .9])
        ax.imshow(image, aspect='auto')
        for j, best_region_id in enumerate(best_region_ids):
            if j >= len(caption):
                break
            bounding_box = bb_id_to_bb_map[best_region_id]
            alignment_score = alignment_scores[j]

            if alignment_score > good_alignment_threshold:
                alignment_color = HEX_COLOR_LIST[int(best_region_id) % len(HEX_COLOR_LIST)]
                ax.add_patch(patches.Rectangle(
                    (bounding_box[0], bounding_box[1]),
                    bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1],
                    fill=False, linewidth=3, edgecolor=alignment_color))
                score_word_text = '%.2f   %s' % (alignment_score, dictionary[caption[j]])
                ax.text(image.shape[1], 20*j, score_word_text, bbox={'facecolor': alignment_color, 'alpha': 0.2})
                # Draw line between rectangle and word
                endpt_1 = [(bounding_box[0] + bounding_box[2])/2, (bounding_box[1] + bounding_box[3])/2]
                endpt_2 = [image.shape[1], 20*j]
                ax.plot([endpt_1[0], endpt_2[0]], [endpt_1[1], endpt_2[1]], color=alignment_color)
            else:
                alignment_color = '#FFFFFF'
                score_word_text = '%.2f   %s' % (alignment_score, dictionary[caption[j]])
                ax.text(image.shape[1], 20*j, score_word_text, bbox={'facecolor': alignment_color, 'alpha': 0.2})

            ax.axis('off')

        plt.draw()
        plt.waitforbuttonpress()
        plt.clf()

if __name__ == '__main__':
    main()
