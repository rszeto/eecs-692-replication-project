import numpy as np
import os
import sys
from gensim.models.keyedvectors import KeyedVectors

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

def main():
    dataset = 'flickr8k'

    ### Get dataset information ###
    if dataset == 'flickr8k':
        dataset_wrapper = Flickr8kFactory().get_dataset_wrapper()
    elif dataset == 'mscoco':
        dataset_wrapper = MSCOCOFactory().get_dataset_wrapper()
    else:
        print('Unknown dataset %s' % dataset)
        return

    dictionary = dataset_wrapper.get_dictionary()
    word_vectors = KeyedVectors.load_word2vec_format('/home/szetor/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    W_w = np.zeros((len(dictionary), 300))
    num_unknown_words = 0
    for i, word in enumerate(dictionary):
        try:
            W_w[i, :] = word_vectors[word]
        except KeyError:
            print('%s not found, replacing with UNK embedding' % word)
            num_unknown_words += 1
            W_w[i, :] = word_vectors['UNK']

    np.save('word2vec_mat', W_w)
    print('Done. Found %d unknown words' % num_unknown_words)


if __name__ == '__main__':
    main()