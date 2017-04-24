import numpy as np

def tile_batches(a):
    '''
    Tiles the input across batches. The first dimension of the result matches the second dimension
    of the input, and the second dimension
    :param a: A numpy array of shape batch_size x rows x cols
    :return: The tiled input. The rows match the second dimension of the input, and the columns
        correspond to the third dimension of the input, except each batch is lined up next to
        each other.
    '''
    return a.transpose([1, 0, 2]).reshape((a.shape[1], a.shape[0] * a.shape[2]))


def image_sentence_score_np(v_i, s_t, caption_length):
    '''
    Computes the image-sentence score.
    :param v_i:
    :param s_t:
    :return:
    '''
    max_caption_length = s_t.shape[1]
    alignment_scores = np.dot(v_i.T, s_t)
    best_alignment_scores = np.max(alignment_scores, axis=0)
    # Remove scores from padded words
    for i in range(caption_length, max_caption_length):
        best_alignment_scores[i] = 0
    return np.sum(best_alignment_scores)


def S_kl_np_slow(batch_region_embeddings, batch_word_embeddings, caption_lengths):
    num_images = batch_region_embeddings.shape[0]
    num_sentences = batch_word_embeddings.shape[0]

    S_kl = np.zeros((num_images, num_sentences))
    for i in range(num_images):
        for j in range(num_sentences):
            S_kl[i, j] = image_sentence_score_np(batch_region_embeddings[i, :, :], batch_word_embeddings[j, :, :], caption_lengths[j])

    return S_kl


def S_kl_np(batch_region_embeddings, batch_word_embeddings, caption_lengths):
    '''
    Compute the cost of a batch of region and word embeddings. Mask the word embeddings
    corresponding to padded words. Verified to be equivalent
    :param batch_region_embeddings:
    :param batch_word_embeddings:
    :param caption_lengths:
    :return:
    '''
    num_images = batch_region_embeddings.shape[0]
    num_sentences = batch_word_embeddings.shape[0]
    num_image_regions = batch_region_embeddings.shape[2]
    max_caption_length = batch_word_embeddings.shape[2]

    tiled_batch_region_embeddings = tile_batches(batch_region_embeddings)
    tiled_batch_word_embeddings = tile_batches(batch_word_embeddings)

    # Create mask over padded words (num_sentences x max_caption_length x num_images)
    lengths_tiled = np.tile(np.array(caption_lengths).reshape([num_sentences, 1, 1]), [1, max_caption_length, num_images])
    iter_tiled = np.tile(np.array(range(max_caption_length)).reshape((1, max_caption_length, 1)), [num_sentences, 1, num_images])
    mask = np.array(iter_tiled < lengths_tiled)

    region_word_alignment_scores = np.dot(tiled_batch_region_embeddings.T, tiled_batch_word_embeddings)
    # Reshape so that rows correspond to words and region scores for words are in columns
    # Rows correspond to (sentence, word, image) order: (0, 0, 0), (0, 0, 1), ... (0, 1, 0), ... (1, 0, 0), ...
    region_word_alignment_scores = region_word_alignment_scores.T.reshape(region_word_alignment_scores.size/num_image_regions, num_image_regions)
    # Get maximum score of each word
    max_word_scores = np.max(region_word_alignment_scores, axis=1)
    # Reshape word scores to get sentence, word, and image dimensions
    max_word_scores = max_word_scores.reshape(batch_word_embeddings.shape[0], max_caption_length, batch_region_embeddings.shape[0])
    # Mask padded words so they don't contribute to score
    max_word_scores *= mask
    # Transpose into image, sentence, word dimensions
    max_word_scores = np.transpose(max_word_scores, [2, 0, 1])
    # Sum over word dimension
    S_kl = np.sum(max_word_scores, axis=2)
    return S_kl
