import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell

# Define hyperparameters
joint_embedding_size = 1000
word_embedding_size = 300
num_hidden_units = 300
batch_size = 100
STD_DEV = 0.1
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
REGULARIZATION_CONSTANT = 0.0001
DROPOUT_KEEP_RATE = 0.5

# Define constant properties
NUM_IMAGE_REGIONS = 20

### BEGIN UTIL FUNCTIONS ###

def read_and_decode(filename_queue, dictionary_size, max_caption_length):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_id': tf.FixedLenFeature([], tf.int64),
            'caption_id': tf.FixedLenFeature([], tf.int64),
            'bounding_box_ids': tf.FixedLenFeature([], tf.string),
            'fc7_activation_set': tf.FixedLenFeature([], tf.string),
            'image_caption': tf.FixedLenFeature([], tf.string)
        }
    )

    # Decode image and caption ids
    image_id = tf.cast(features['image_id'], tf.int32)
    caption_id = tf.cast(features['caption_id'], tf.int32)
    # Decode byte lists
    bounding_box_ids = tf.decode_raw(features['bounding_box_ids'], tf.int32)
    fc7_activation_set = tf.decode_raw(features['fc7_activation_set'], tf.float32)
    image_caption = tf.decode_raw(features['image_caption'], tf.int32)
    # Cast image_caption to float32 for compatibility with RNN
    image_caption = tf.cast(image_caption, tf.float32)
    bounding_box_ids.set_shape((NUM_IMAGE_REGIONS))
    fc7_activation_set.set_shape((4096 * NUM_IMAGE_REGIONS))
    fc7_activation_set = tf.reshape(fc7_activation_set, (4096, NUM_IMAGE_REGIONS))
    # image_caption.set_shape([dictionary_size, max_caption_length])
    image_caption.set_shape((dictionary_size * 36))
    image_caption = tf.reshape(image_caption, (dictionary_size, 36))
    image_caption = tf.slice(image_caption, [0, 0], [dictionary_size, max_caption_length])

    return image_id, caption_id, bounding_box_ids, fc7_activation_set, image_caption


def batch_matmul(a, b, shape=None):
    '''
    Compute each term a_i * b, where a_i indexes each tensor in the batch. For 2D matrices
    a_i and b, the second dimension of a_i must match the first dimension of b.
    Source: http://stackoverflow.com/a/41986618
    :param a:
    :param b:
    :param shape: A list denoting the final shape of the output. Used for easier debugging.
    :return:
    '''
    final_shape = tf.concat((tf.shape(a)[:-1], tf.shape(b)[-1:]), 0)
    a_reshaped = tf.reshape(a, (tf.reduce_prod(tf.shape(a)[:-1]), tf.shape(b)[0]))
    ab_reshaped = tf.matmul(a_reshaped, b)
    ret = tf.reshape(ab_reshaped, final_shape)
    if shape:
        ret.set_shape(shape)
    return ret


def caption_length(caption):
    '''
    Compute true caption length from a padded Tensor representing the caption's one-hot encodings.
    :param caption:
    :return:
    '''
    # Sum first along the dictionary dim, then the "sentence time" dim
    return tf.cast(tf.reduce_sum(tf.reduce_sum(caption, 1), 1), tf.int32)


def batch_image_captions_to_seq_mask(batch_image_captions, output_dim_size):
    '''
    Compute a mask with 1s for values that should be kept and 0s elsewhere.
    :param batch_image_captions: The batched image caption tensor
    :param output_dim_size: The second dimension of the output (how large your features are)
    :return:
    '''
    single_feature_mask = tf.expand_dims(tf.reduce_sum(batch_image_captions, 1), 1)
    return tf.tile(single_feature_mask, [1, output_dim_size, 1])


def image_sentence_score(v_i, s_t, caption_length, max_caption_length):
    '''
    Computes the image-sentence score S_kl (Eq. 8)
    :param v_i: Image embedding with dimensions embedding_size x NUM_IMAGE_REGIONS
    :param s_t: Word embedding with dimensions embedding_size x max_cap_len
              Embeddings past the caption length should be zeros.
    :param caption_length: How long the given caption is.
    :return:
    '''
    v_T = tf.transpose(v_i, perm=[1, 0])
    alignment_mat = tf.matmul(v_T, s_t)
    alignment_scores = tf.reduce_max(alignment_mat, axis=0)
    # Mask padded words
    mask = tf.cast(tf.less(tf.range(0, max_caption_length), caption_length), tf.float32)
    score = tf.reduce_sum(alignment_scores * mask)
    return score


# def dumb_embedding_cost(v, s):
#     '''
#     Compute some meaningless cost on the given embeddings. Used to prove that an efficient objective
#     is needed to fit a large batch on the GPU.
#     :param v: Image embeddings with dimensions batch_size x embedding_size x NUM_IMAGE_REGIONS
#     :param s: Word embeddings with dimensions batch_size x embedding_size x max_cap_len.
#               Embeddings past the caption length should be zeros.
#     :return:
#     '''
#     reshaped_v_i = tf.reshape(v, [batch_size * joint_embedding_size, NUM_IMAGE_REGIONS])
#     reshaped_s_t = tf.reshape(s, [batch_size * joint_embedding_size, max_caption_length])
#     prod = tf.matmul(tf.transpose(reshaped_v_i, perm=[1, 0]), reshaped_s_t)
#     return tf.reduce_sum(tf.reduce_sum(prod))


def embedding_cost(v, s, max_caption_length, caption_lengths, batch_size, num_image_regions, joint_embedding_size):
    '''
    Computes the embedding portion of the loss for a batch (Eq. 9)
    :param v: Image embeddings with dimensions batch_size x embedding_size x NUM_IMAGE_REGIONS
    :param s: Word embeddings with dimensions batch_size x embedding_size x max_cap_len.
              Embeddings past the caption length should be zeros.
    :return:
    '''
    S_kl = compute_S_kl(v, s, caption_lengths)

    # "Rank images" part
    S_kk = tf.diag_part(S_kl)
    S_kk_tile_across_l = tf.tile(tf.reshape(S_kk, [batch_size, 1]), [1, batch_size])
    rank_images = tf.reduce_sum(tf.maximum(0.0, S_kl - S_kk_tile_across_l + 1), axis=1)

    # "Rank sentences" part
    S_kk_tile_across_k = tf.tile(tf.reshape(S_kk, [1, batch_size]), [batch_size, 1])
    rank_sentences = tf.reduce_sum(tf.maximum(0.0, S_kl - S_kk_tile_across_k + 1), axis=0)

    # Sum over all examples in batch
    final_cost = tf.reduce_sum(rank_images + rank_sentences) / batch_size ** 2
    return final_cost, S_kl


def tile_batches(a):
    '''
    Tiles the input across batches.
    :param a: A tf.Tensor of shape batch_size x rows x cols
    :return: The tiled input. The rows match the second dimension of the input, and the columns
        correspond to the third dimension of the input, except each batch is lined up next to
        each other.
    '''
    a_shape = tf.shape(a)
    return tf.reshape(tf.transpose(a, perm=[1, 0, 2]), ([a_shape[1], a_shape[0] * a_shape[2]]))


def compute_S_kl(v, s, caption_lengths):
    v_shape = tf.shape(v)
    s_shape = tf.shape(s)
    num_images = v_shape[0]
    num_image_regions = v_shape[2]
    num_captions = s_shape[0]
    max_caption_length = s_shape[2]

    tiled_batch_region_embeddings = tile_batches(v)
    tiled_batch_word_embeddings = tile_batches(s)

    # Create mask over padded words (shape is num_sentences x max_caption_length x num_images)
    mask = tf.cast(tf.tile(tf.expand_dims(tf.sequence_mask(caption_lengths, max_caption_length), -1), [1, 1, num_images]), tf.float32)

    region_word_alignment_scores = tf.matmul(tf.transpose(tiled_batch_region_embeddings), tiled_batch_word_embeddings)
    # Reshape so that rows correspond to words and region scores for words are in columns
    # Rows correspond to (sentence, word, image) order: (0, 0, 0), (0, 0, 1), ... (0, 1, 0), ... (1, 0, 0), ...
    region_word_alignment_scores = tf.reshape(tf.transpose(region_word_alignment_scores), [tf.size(region_word_alignment_scores) / num_image_regions, num_image_regions])
    # Get maximum score of each word
    max_word_scores = tf.reduce_max(region_word_alignment_scores, axis=1)
    # Reshape to get sentence, word, and image dimensions
    max_word_scores = tf.reshape(max_word_scores, [num_captions, max_caption_length, num_images])
    # Mask padded words
    max_word_scores *= mask
    # Transpose into image, sentence, word dimensions
    max_word_scores = tf.transpose(max_word_scores, [2, 0, 1])
    # Sum over word dimension
    S_kl = tf.reduce_sum(max_word_scores, axis=2)
    return S_kl


def regularization_cost(W_list):
    '''
    Computes the unweighted regularization term of the loss.
    :param W_list: The list of weights to learn/regularize.
    :return:
    '''
    return tf.reduce_sum([tf.nn.l2_loss(W) for W in W_list])

### END UTIL FUNCTIONS


class RegionCaptionDataLoader:

    def __init__(self, record_path, dictionary_size, max_caption_length, shuffle=True):
        '''
        Constructor
        :param record_path: Path to a .tfrecords file containing paired images (as a set of fc7
            activations) and sentences (as a set of one-hot encodings for each word)
        '''
        filename_queue = tf.train.string_input_producer([record_path])
        image_id, caption_id, bounding_box_ids, fc7_activation_set, image_caption = read_and_decode(filename_queue, dictionary_size, max_caption_length)

        if shuffle:
            self._batch_image_ids, self._batch_caption_ids, self._batch_bounding_box_ids, self._batch_fc7_activation_sets, self._batch_image_captions = tf.train.shuffle_batch(
                [image_id, caption_id, bounding_box_ids, fc7_activation_set, image_caption],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000
            )
        else:
            data_queue = tf.FIFOQueue(
                capacity=1000 + 3 * batch_size,
                dtypes=[tf.int32, tf.int32, tf.int32, tf.float32, tf.float32],
                shapes=[(), (), (NUM_IMAGE_REGIONS), (4096, NUM_IMAGE_REGIONS), (dictionary_size, max_caption_length)]
            )
            enqueue_data_op = data_queue.enqueue([image_id, caption_id, bounding_box_ids, fc7_activation_set, image_caption])
            # Add to global queue runner list
            qr = tf.train.QueueRunner(data_queue, [enqueue_data_op] * 2)
            tf.train.add_queue_runner(qr)
            self._batch_image_ids, self._batch_caption_ids, self._batch_bounding_box_ids, self._batch_fc7_activation_sets, self._batch_image_captions = data_queue.dequeue_many(batch_size)


    def get_batch_image_ids(self):
        return self._batch_image_ids

    def get_batch_caption_ids(self):
        return self._batch_caption_ids

    def get_batch_fc7_activation_sets(self):
        return self._batch_fc7_activation_sets

    def get_batch_image_captions(self):
        return self._batch_image_captions

    def get_batch_bounding_box_ids(self):
        return self._batch_bounding_box_ids


class AlignmentModel:

    def __init__(self, batch_fc7_activation_sets, batch_image_captions, dictionary_size, max_caption_length, test_only=False, use_word2vec=True):
        '''
        Constructor
        :param batch_fc7_activation_sets: Tensor representing a batch of fc7 region features.
            Shape should be batch_size x 4096 x 20.
        :param batch_image_captions: Tensor representing a batch of sentences, where each word
            is a one-hot encoding. Shape should be batch_size x dictionary_len x max_caption_len.
        '''

        # Placeholder to modify dropout during train/test time
        self._dropout_keep_rate = tf.placeholder(tf.float32)

        # Compute image embedding (Eq. 1)
        batch_fc7_activation_sets_t = tf.transpose(batch_fc7_activation_sets, perm=[0, 2, 1])
        W_m = tf.Variable(tf.truncated_normal([4096, joint_embedding_size], stddev=0.01), name='W_m')
        b_m = tf.Variable(tf.truncated_normal([1, joint_embedding_size, 1], stddev=0.01), name='b_m')
        # Tile b_m across batches and image regions
        b_m_tile = tf.tile(b_m, tf.stack([batch_size, 1, NUM_IMAGE_REGIONS]))
        # Compute Wx
        v_i_a = batch_matmul(batch_fc7_activation_sets_t, W_m, shape=[batch_size, NUM_IMAGE_REGIONS, joint_embedding_size])
        # Compute Wx + b
        v_i_b = tf.transpose(v_i_a, perm=[0, 2, 1])
        v_i = v_i_b + b_m_tile
        # Dropout
        v_i = tf.nn.dropout(v_i, self._dropout_keep_rate)
        self._region_embeddings = v_i

        # Word projection (Eq. 2)
        # Transpose so all words are multiplied by the same (context-independent) word embedding
        batch_image_captions_t = tf.transpose(batch_image_captions, perm=[0, 2, 1])
        if use_word2vec:
            # Load word2vec embeddings
            word2vec_mat = np.load('word2vec_mat.npy')
            W_w = tf.constant(word2vec_mat, dtype=tf.float32)
        else:
            W_w = tf.Variable(tf.truncated_normal([dictionary_size, word_embedding_size], stddev=100 * STD_DEV), name='W_w')
        x_t = batch_matmul(batch_image_captions_t, W_w, shape=[batch_size, max_caption_length, word_embedding_size])
        # Dropout
        x_t = tf.nn.dropout(x_t, self._dropout_keep_rate)

        # Input to RNN cells (Eq. 3)
        W_e = tf.Variable(tf.truncated_normal([word_embedding_size, num_hidden_units], stddev=STD_DEV), name='W_e')
        # W_e = tf.Variable(tf.truncated_normal([dictionary_size, num_hidden_units], stddev=STD_DEV), name='W_e')
        b_e = tf.Variable(tf.truncated_normal([1, num_hidden_units, 1], stddev=STD_DEV), name='b_e')
        # Tile b_e across batches and time steps
        b_e_tile = tf.tile(b_e, tf.stack([batch_size, 1, max_caption_length]))
        # Compute Wx
        e_t_a = batch_matmul(x_t, W_e, shape=[batch_size, max_caption_length, num_hidden_units])
        # e_t_a = batch_matmul(batch_image_captions_t, W_e, shape=[batch_size, max_caption_length, num_hidden_units])
        # Compute Wx + b
        e_t_b = tf.transpose(e_t_a, perm=[0, 2, 1])
        e_t_c = e_t_b + b_e_tile
        # Compute f(Wx + b)
        e_t = tf.nn.relu(e_t_c)
        # Dropout
        e_t = tf.nn.dropout(e_t, self._dropout_keep_rate)
        # Transpose e_t so time is second dimension (needed for RNN interface)
        e_t = tf.transpose(e_t, perm=[0, 2, 1])
        caption_lengths = caption_length(batch_image_captions)

        # Forward RNN state (Eq. 4)
        forward_rnn_cell = BasicRNNCell(num_hidden_units, activation=tf.nn.relu)
        htf, _ = tf.nn.dynamic_rnn(
            forward_rnn_cell,
            e_t,
            dtype=tf.float32,
            sequence_length=caption_lengths,
            scope='forward_rnn_cell'
        )

        # Backward RNN state (Eq. 5)
        backward_rnn_cell = BasicRNNCell(num_hidden_units, activation=tf.nn.relu)
        e_t_rev = tf.reverse_sequence(e_t, caption_lengths, batch_axis=0, seq_axis=1)
        htb, _ = tf.nn.dynamic_rnn(
            backward_rnn_cell,
            e_t_rev,
            dtype=tf.float32,
            sequence_length=caption_lengths,
            scope='backward_rnn_cell'
        )
        # Reverse htb so time index goes forward
        htb = tf.reverse_sequence(htb, caption_lengths, batch_axis=0, seq_axis=1)

        # Get final word embeddings (Eq. 6)
        W_d = tf.Variable(tf.truncated_normal([num_hidden_units, joint_embedding_size], stddev=STD_DEV), name='W_d')
        b_d = tf.Variable(tf.truncated_normal([1, joint_embedding_size, 1], stddev=STD_DEV), name='b_d')
        # Tile b_d across batches and time steps
        b_d_tile = tf.tile(b_d, tf.stack([batch_size, 1, max_caption_length]))
        # Compute Wx
        ht_sum = htf + htb
        s_t_a = batch_matmul(ht_sum, W_d, shape=[batch_size, max_caption_length, joint_embedding_size])
        s_t_b = tf.transpose(s_t_a, perm=[0, 2, 1])
        # Compute Wx + b
        s_t_c = s_t_b + b_d_tile
        # Compute f(Wx + b)
        s_t = tf.nn.relu(s_t_c)
        # Dropout
        s_t = tf.nn.dropout(s_t, self._dropout_keep_rate)

        # # Go from word2vec to final embedding in one go
        # W_d = tf.Variable(tf.truncated_normal([word_embedding_size, joint_embedding_size], stddev=STD_DEV), name='W_d')
        # b_d = tf.Variable(tf.truncated_normal([1, joint_embedding_size, 1], stddev=STD_DEV), name='b_d')
        # # Tile b_d across batches and image regions
        # b_d_tile = tf.tile(b_d, tf.stack([batch_size, 1, max_caption_length]))
        # # Compute f(Wx + b)
        # s_t_a = batch_matmul(x_t, W_d, shape=[batch_size, max_caption_length, joint_embedding_size])
        # s_t_b = tf.transpose(s_t_a, perm=[0, 2, 1])
        # s_t_c = s_t_b + b_d_tile
        # s_t = tf.nn.relu(s_t_c)
        # # Dropout
        # s_t = tf.nn.dropout(s_t, self._dropout_keep_rate)
        # # Define caption lengths
        # caption_lengths = caption_length(batch_image_captions)

        # Zero out embeddings for pad words
        mask = batch_image_captions_to_seq_mask(batch_image_captions, joint_embedding_size)
        mask = tf.cast(mask, s_t.dtype)
        s_t_filtered = tf.multiply(s_t, mask)
        self._word_embeddings = s_t_filtered

        # Compute loss (Eq. 9)
        if not test_only:
            self._region_embedding_norms = tf.norm(tf.reshape(tf.transpose(v_i, perm=[0, 2, 1]), [batch_size * 20, joint_embedding_size]), axis=1)
            self._word_embedding_norms = tf.norm(tf.reshape(tf.transpose(s_t_filtered, perm=[0, 2, 1]), [batch_size * max_caption_length, joint_embedding_size]), axis=1)

            embed_cost, self._S_kl = embedding_cost(v_i, s_t_filtered, max_caption_length, caption_lengths, batch_size, NUM_IMAGE_REGIONS, joint_embedding_size)
            weights = [W for W in tf.trainable_variables() if 'W_' in W.name or 'weights' in W.name]
            reg_cost = regularization_cost(weights)
            self._loss = embed_cost + REGULARIZATION_CONSTANT * reg_cost
            # Optimize over loss
            optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
            gradients = optimizer.compute_gradients(self._loss)
            clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients]
            self._train_op = optimizer.apply_gradients(clipped_gradients)
            # self._train_op = optimizer.apply_gradients(gradients)


    def get_region_embeddings(self):
        return self._region_embeddings

    def get_word_embeddings(self):
        return self._word_embeddings

    def get_region_embedding_norms(self):
        return self._region_embedding_norms

    def get_word_embedding_norms(self):
        return self._word_embedding_norms

    def get_S_kl_scores(self):
        return self._S_kl

    def compute_loss(self, sess):
        return sess.run(self._loss, feed_dict={self._dropout_keep_rate: 1.0})

    def step(self, sess):
        return sess.run(self._train_op, feed_dict={self._dropout_keep_rate: DROPOUT_KEEP_RATE})

    def run_inference(self, sess, ops):
        '''
        Compute the values of the given ops at inference time (i.e. no dropout)
        :param sess: The tensorflow session.
        :param ops: A list of tensors to compute.
        :return:
        '''
        return sess.run(ops, feed_dict={self._dropout_keep_rate: 1.0})