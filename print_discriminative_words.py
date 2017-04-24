import numpy as np
import os
import sys

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

def main():
    dataset = 'flickr8k'
    eval_split = 'test'
    training_run_name = 'Sun_Apr_23_12:33:53_2017'
    snapshot_iter_num = 20000

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
    max_caption_length = dataset_wrapper.get_max_caption_length()
    caption_data = eval_data['caption_data']

    # Load image and caption IDs and embeddings
    print('Loading saved IDs and embeddings')
    results_path = os.path.join(SCRIPT_DIR, 'eval_results', '%s_%s_%s_%s' % (dataset, eval_split, training_run_name, snapshot_iter_num))
    all_caption_ids = np.load(os.path.join(results_path, 'caption_ids.npy'))
    all_word_embeddings = np.load(os.path.join(results_path, 'word_embeddings.npy'))

    # Create index from caption IDs to padded arrays of word IDs
    caption_to_word_id_arr_map = {}
    for datum in caption_data:
        caption_id = datum['caption_id']
        word_id_list = datum['caption']
        word_id_arr = -1 * np.ones(max_caption_length)
        word_id_arr[:len(word_id_list)] = word_id_list
        caption_to_word_id_arr_map[caption_id] = word_id_arr

    # Create array that identifies each word in the word embedding matrix
    word_ids = np.zeros((len(all_caption_ids), max_caption_length))
    for i, caption_id in enumerate(all_caption_ids):
        word_ids[i, :] = caption_to_word_id_arr_map[caption_id]

    # Get average magnitude of each word in the dictionary
    avg_word_magnitudes = np.zeros(len(dictionary))
    for i in range(len(dictionary)):
        cur_word_indexes = np.where(word_ids == i)
        word_embeddings = all_word_embeddings[cur_word_indexes[0], :, cur_word_indexes[1]]
        if word_embeddings.size == 0:
            pass
        avg_word_magnitudes[i] = np.linalg.norm(np.mean(word_embeddings))

    # Get sorted indexes over word magnitudes
    sorted_magnitude_indexes = np.argsort(avg_word_magnitudes)
    # Cut out indexes corresponding to nan (i.e. word didn't occur)
    sorted_magnitudes = np.sort(avg_word_magnitudes)
    first_nan_index = np.where(np.isnan(sorted_magnitudes))[0][0]
    sorted_magnitude_indexes = sorted_magnitude_indexes[:first_nan_index]

    # Print table of magnitudes (Table 2)
    smallest_magnitude_indexes = sorted_magnitude_indexes[:20]
    largest_magnitude_indexes = sorted_magnitude_indexes[-20:][::-1]
    print('Magnitude\tWord\t\t\tMagnitude\tWord')
    for i in range(20):
        print('%.10f\t%s\t\t\t\t%.10f\t%s' % (
            avg_word_magnitudes[smallest_magnitude_indexes[i]],
            dictionary[smallest_magnitude_indexes[i]],
            avg_word_magnitudes[largest_magnitude_indexes[i]],
            dictionary[largest_magnitude_indexes[i]],
        ))

if __name__ == '__main__':
    main()