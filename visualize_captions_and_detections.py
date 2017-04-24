import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
from scipy.misc import imread

sys.path.append(os.path.join('..', 'datasets', 'mscoco'))
from mscoco import MSCOCOFactory
sys.path.append(os.path.join('..', 'datasets', 'flickr8k'))
from flickr8k import Flickr8kFactory

if __name__ == '__main__':
    dataset_wrapper_factory = Flickr8kFactory()
    dataset_wrapper = dataset_wrapper_factory.get_dataset_wrapper()
    train_data = dataset_wrapper.get_train_data()
    image_data = train_data['image_data']
    caption_data = train_data['caption_data']
    bounding_box_data = train_data['bounding_box_data']
    dictionary = dataset_wrapper.get_dictionary()

    # # Index the bounding boxes by image
    # bounding_box_index = {datum['image_id']: datum['bounding_boxes'] for datum in bounding_box_data}

    # Index the bounding boxes by image
    bounding_box_index = {}
    for datum in bounding_box_data:
        image_id = datum['image_id']
        if image_id not in bounding_box_index:
            bounding_box_index[image_id] = []
        bounding_box_index[image_id].append(datum)

    # Index the captions by image
    caption_index = {}
    for datum in caption_data:
        image_id = datum['image_id']
        if image_id not in caption_index:
            caption_index[image_id] = []
        caption_index[image_id].append(datum)

    fig = plt.figure()

    # Go through images
    for image in image_data:
        image_id = image['image_id']
        file_path = image['file_path']

        # Show image
        im = imread(file_path)
        plt.imshow(im)

        # Show bounding boxes
        ax = fig.add_subplot(111)
        bounding_boxes = bounding_box_index[image_id]
        for datum in bounding_boxes:
            bounding_box = datum['bounding_box']
            ax.add_patch(patches.Rectangle(
                (bounding_box[0], bounding_box[1]),
                bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1],
                fill=False, linewidth=3, edgecolor='blue'))

        # Show captions
        caption_datas = caption_index[image_id]
        suptitle = ''
        for datum in caption_datas:
            for word in datum['caption']:
                suptitle += '<%s> ' % dictionary[word]
            suptitle += '\n'
        plt.suptitle(suptitle)

        plt.subplots_adjust(top=0.75)
        plt.draw()
        plt.waitforbuttonpress()
        plt.clf()