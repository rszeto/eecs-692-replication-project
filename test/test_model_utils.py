import numpy as np
import tensorflow as tf
import os
import sys

# Add code directory to Python path
TEST_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(TEST_DIR, '..'))
from eval_utils import image_sentence_score_np
from eval_utils import S_kl_np
from eval_utils import S_kl_np_slow

ERROR_THRESHOLD = 1e-6

def assert_match(gt_val, val):
    '''
    Check that the given value is epsilon within the ground truth
    :param gt_val:
    :param val:
    :return:
    '''
    if np.max(np.abs(gt_val - val)) > ERROR_THRESHOLD:
        raise AssertionError('Result %f does not match actual value %f' % (val, gt_val))


def test_image_sentence_score_model_utils():
    from model_utils import image_sentence_score
    np.random.seed(123)

    embedding_size = 5
    caption_length = 4
    num_regions = 3
    max_caption_length = 5

    v_i = np.random.rand(embedding_size, num_regions)
    s_t = np.random.rand(embedding_size, max_caption_length)
    gt_score = image_sentence_score_np(v_i, s_t, caption_length)

    v_i_tf = tf.constant(v_i, dtype=tf.float32)
    s_t_tf = tf.constant(s_t, dtype=tf.float32)
    caption_length_tf = tf.constant(caption_length, dtype=tf.int32)
    score_tf = image_sentence_score(v_i_tf, s_t_tf, caption_length_tf, max_caption_length)

    with tf.Session() as sess:
        score = sess.run(score_tf)

    assert_match(gt_score, score)


def test_S_kl_np():
    np.random.seed(123)

    batch_size = 5
    embedding_size = 5
    num_regions = 4
    max_caption_length = 10
    caption_lengths = [5, 6, 3, 3, 5]

    v = np.random.rand(batch_size, embedding_size, num_regions)
    s = np.random.rand(batch_size, embedding_size, max_caption_length)
    gt_S_kl = S_kl_np_slow(v, s, caption_lengths)
    S_kl = S_kl_np(v, s, caption_lengths)

    assert_match(gt_S_kl, S_kl)


def test_tile_batches():
    from eval_utils import tile_batches as tile_batches_np
    from model_utils import tile_batches as tile_batches_tf
    np.random.seed(123)

    batch_size = 5
    embedding_size = 5
    num_regions = 4

    v = np.random.rand(batch_size, embedding_size, num_regions)
    gt_res = tile_batches_np(v)

    v_tf = tf.constant(v, dtype=tf.float32)
    tiled_batches = tile_batches_tf(v_tf)

    with tf.Session() as sess:
        res = sess.run(tiled_batches)

    assert_match(gt_res, res)


def test_compute_S_kl():
    from model_utils import compute_S_kl
    np.random.seed(123)

    batch_size = 5
    embedding_size = 5
    num_regions = 4
    max_caption_length = 10
    caption_lengths = [5, 6, 3, 3, 5]

    v = np.random.rand(batch_size, embedding_size, num_regions)
    s = np.random.rand(batch_size, embedding_size, max_caption_length)
    gt_S_kl = S_kl_np_slow(v, s, caption_lengths)

    v_tf = tf.constant(v, dtype=tf.float32)
    s_tf = tf.constant(s, dtype=tf.float32)
    S_kl_tf = compute_S_kl(v_tf, s_tf, caption_lengths)

    with tf.Session() as sess:
        S_kl = sess.run(S_kl_tf)

    assert_match(gt_S_kl, S_kl)


def test_embedding_cost():
    from model_utils import embedding_cost
    np.random.seed(123)

    batch_size = 5
    embedding_size = 5
    num_regions = 4
    max_caption_length = 10
    caption_lengths = [5, 6, 3, 3, 5]
    assert(len(caption_lengths) == batch_size)

    v = np.random.rand(batch_size, embedding_size, num_regions)
    s = np.random.rand(batch_size, embedding_size, max_caption_length)

    S = S_kl_np_slow(v, s, caption_lengths)

    gt_cost = 0.0
    for k in range(batch_size):
        for l in range(batch_size):
            gt_cost += np.max([0, S[k, l] - S[k, k] + 1])
        for l in range(batch_size):
            gt_cost += np.max([0, S[l, k] - S[k, k] + 1])
    gt_cost /= batch_size ** 2

    v_tf = tf.constant(v, dtype=tf.float32)
    s_tf = tf.constant(s, dtype=tf.float32)
    caption_lengths_tf = tf.constant(caption_lengths, dtype=tf.int32)
    cost_tf, _ = embedding_cost(v_tf, s_tf, max_caption_length, caption_lengths_tf, batch_size, num_regions, embedding_size)

    with tf.Session() as sess:
        cost = sess.run(cost_tf)

    assert_match(gt_cost, cost)


if __name__ == '__main__':
    test_image_sentence_score_model_utils()
    test_S_kl_np()
    test_embedding_cost()
    test_tile_batches()
    test_compute_S_kl()


    print('All tests passed')