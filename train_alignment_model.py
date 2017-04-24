import numpy as np
import tensorflow as tf
import os
import sys
import time
import glob
import shutil
from model_utils import AlignmentModel
from model_utils import RegionCaptionDataLoader

# Import dataset wrappers
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

PRINT_RATE = 20
SAVE_RATE = 2000
NUM_ITERS = 200000


def main():

    dataset = 'flickr8k'
    data_split = 'train'
    use_one_caption_per_image = False
    resume_run_name = None
    # resume_run_name = 'Sun_Apr_23_20:30:29_2017'
    comment = ''

    ### Get dataset information ###
    if dataset == 'flickr8k':
        dataset_wrapper = Flickr8kFactory().get_dataset_wrapper()
    elif dataset == 'mscoco':
        dataset_wrapper = MSCOCOFactory().get_dataset_wrapper()
    else:
        print('Unknown dataset %s' % dataset)
        return

    if use_one_caption_per_image:
        data_split += '_one_caption'
    record_path = os.path.join(SCRIPT_DIR, 'paired_data', '%s_%s.tfrecords' % (dataset, data_split))
    dictionary = dataset_wrapper.get_dictionary()
    max_caption_length = dataset_wrapper.get_max_caption_length()

    ### Define data loader and model pipeline
    data_loader = RegionCaptionDataLoader(record_path, len(dictionary), max_caption_length)
    batch_fc7_activation_sets = data_loader.get_batch_fc7_activation_sets()
    batch_image_captions = data_loader.get_batch_image_captions()
    model = AlignmentModel(batch_fc7_activation_sets, batch_image_captions, len(dictionary), max_caption_length)

    # Print number of unique images in batch
    batch_image_ids = data_loader.get_batch_image_ids()
    unique_ids, _ = tf.unique(batch_image_ids)
    print_percent_unique_image_ids = tf.Print(batch_image_ids, [tf.size(unique_ids), tf.shape(batch_image_ids)], message='Num unique image IDs and batch size: ')

    # Weight initializer
    init_op = tf.global_variables_initializer()

    # Save checkpoints with saver
    saver = tf.train.Saver(max_to_keep=1000)

    # Create folder for saving checkpoints and losses for current run
    if resume_run_name:
        print('Resuming from training run started at %s' % resume_run_name)
    else:
        print('Creating new folder for current training run')
    dataset_split_key = '%s_%s' % (dataset, data_split)
    date_time_key = resume_run_name if resume_run_name else time.strftime('%c').replace(' ', '_')
    training_run_root = os.path.join(SCRIPT_DIR, 'training_runs', dataset_split_key, date_time_key)
    if not os.path.exists(training_run_root):
        os.makedirs(training_run_root)
    # Define checkpoint base (includes path and first part of checkpoint identifier)
    checkpoint_base = os.path.join(training_run_root, 'checkpoint')

    # Define and open loss file
    num_loss_files = glob.glob(os.path.join(training_run_root, 'loss*.csv'))
    loss_file_path = os.path.join(training_run_root, 'loss_%d.csv' % len(num_loss_files))
    loss_file = open(loss_file_path, 'w')
    loss_file.write('Iteration,Loss\n')
    # Define and open file for other info (embedding norms and S_kl)
    num_aux_files = glob.glob(os.path.join(training_run_root, 'aux*.csv'))
    aux_file_path = os.path.join(training_run_root, 'aux_%d.csv' % len(num_aux_files))
    aux_file = open(aux_file_path, 'w')
    aux_file.write(',Iteration,Min,Max,Mean_Trace\n')
    # Copy model_utils file for documentation
    model_utils_path = os.path.join(SCRIPT_DIR, 'model_utils.py')
    shutil.copy(model_utils_path, training_run_root)
    # Write comment in README.md
    with open(os.path.join(training_run_root, 'README.md'), 'a') as f:
        f.write('%s\n' % comment)

    with tf.Session() as sess:
        if resume_run_name:
            last_checkpoint_path = tf.train.latest_checkpoint(training_run_root)
            resume_iter = int(last_checkpoint_path.replace(checkpoint_base + '-', ''))
            saver.restore(sess, checkpoint_base + '-' + str(resume_iter))
        else:
            resume_iter = 0
            # Initialize weights
            sess.run(init_op)

        # Start implicit queue from read_and_decode
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Started training')
        for i in range(resume_iter, NUM_ITERS + 1):
            # Print and write current loss
            if i % PRINT_RATE == 0:
                # loss_val = model.compute_loss(sess)
                loss_val, _ = model.run_inference(sess, [model._loss, print_percent_unique_image_ids])
                print('Iteration = %d, loss = %.5f' % (i, loss_val))
                loss_file.write('%d,%.5f\n' % (i, loss_val))
                loss_file.flush()

                # Print aux info
                region_norms, word_norms, S_kl = model.run_inference(sess, [model.get_region_embeddings(), model.get_word_embeddings(), model.get_S_kl_scores()])
                print('\tRegion norms: %.5f, %.5f' % (np.min(region_norms), np.max(region_norms)))
                print('\tWord norms: %.5f, %.5f' % (np.min(word_norms), np.max(word_norms)))
                print('\tS_kl: %.5f, %.5f, %.5f' % (np.min(S_kl), np.max(S_kl), np.trace(S_kl)/S_kl.shape[0]))
                aux_file.write('%s,%d,%f,%f\n' % ('region_norms', i, np.min(region_norms), np.max(region_norms)))
                aux_file.write('%s,%d,%f,%f\n' % ('word_norms', i, np.min(word_norms), np.max(word_norms)))
                aux_file.write('%s,%d,%f,%f,%f\n' % ('S_kl', i, np.min(S_kl), np.max(S_kl), np.trace(S_kl)/S_kl.shape[0]))
                aux_file.flush()

            # Save checkpoint
            # if i > resume_iter and i % SAVE_RATE == 0:
            if i % SAVE_RATE == 0:
                print('Saving iteration %d... ' % i),
                saver.save(sess, checkpoint_base, global_step=i)
                print('done')

            # Step through optimization
            model.step(sess)

        # Stop threads
        coord.request_stop()
        coord.join(threads)

    print('Finished training')
    loss_file.close()

if __name__ == '__main__':
    main()