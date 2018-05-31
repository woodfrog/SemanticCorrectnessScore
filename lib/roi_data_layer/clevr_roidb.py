"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL.Image
import pdb
import os
import os.path as osp
import pickle
from lib.tree import Tree
import unicodedata


def prepare_clevr_roidb(base_dir, image_dir, tree_dir, detector_type, replace_image_dir=False, new_image_dir=None):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """

    roidb = []

    image_list = sorted(os.listdir(image_dir))

    tree_path_list = [osp.join(tree_dir, filename) for filename in sorted(os.listdir(tree_dir))]
    tree_list = []
    for tree_path in tree_path_list:
        with open(tree_path, 'rb') as f:
            tree_list.append(pickle.load(f))

    if replace_image_dir:
        print('replacing image dir with {}'.format(new_image_dir))
        assert new_image_dir is not None
        new_image_list = list()
        for i in range(
                len(tree_list)):  # to guarantee that the tree list and the image list are consistent with each other
            new_image_list.append(osp.join(new_image_dir, 'image{:05d}.png'.format(i)))
        try:
            assert len(new_image_list) == len(image_list)
        except AssertionError:
            print('the new image list has different length of the standard image list, leading to potential unmatching')
        image_list = new_image_list

    count = 0
    vocabulary = get_vocabulary(tree_list, vocab_dir=base_dir, read_tree=False)
    vocabulary = ['__background__'] + vocabulary

    if detector_type == 'object_attr':
        vocabulary_dict = clevr_obj_attr_vocab()
        # need to change the vocabulary
        obj_attr_vocabulary = vocabulary_dict['obj-attr-comb']

    for im, tree in zip(image_list, tree_list):
        roidb.append({})
        roidb[count]['img_id'] = count
        roidb[count]['image'] = osp.join(image_dir, image_list[count])
        sizes = PIL.Image.open(osp.join(image_dir, image_list[count])).size
        roidb[count]['width'] = sizes[0]
        roidb[count]['height'] = sizes[1]
        # need gt_overlaps as a dense array for argmax
        roidb[count]['boxes'] = get_bbox_from_tree(tree, vocabulary, read_tree=False)
        # print('gt classes, ', np.array(roidb[count]['boxes']))
        overlaps = np.zeros((roidb[count]['boxes'].shape[0], len(vocabulary)), dtype=np.float32)
        roidb[count]['gt_overlaps'] = assign_bbox_overlaps(overlaps, roidb[count]['boxes'])
        gt_overlaps = roidb[count]['gt_overlaps']
        if detector_type == 'object_class':
            roidb[count]['gt_classes'] = np.array(roidb[count]['boxes'])[:, -1]
        elif detector_type == 'objectness':
            gt_classes = np.array(roidb[count]['boxes'])[:, -1]
            gt_classes[:] = 1  # if there is an object, then label is as 1
            roidb[count]['gt_classes'] = gt_classes
        elif detector_type == 'object_attr':
            obj_attr_gt_classes, _ = get_objattr_from_tree(tree, vocabulary_dict, read_tree=False)
            roidb[count]['gt_classes'] = np.array(obj_attr_gt_classes)
        else:
            raise ValueError('Invalid detector type {}, should be one of objectness, object_class, object_attr!'.format(
                detector_type))
        roidb[count]['boxes'] = np.array(roidb[count]['boxes'])[:, :4]
        assert (roidb[count]['boxes'].shape[0] > 0)

        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[count]['max_classes'] = max_classes
        roidb[count]['max_overlaps'] = max_overlaps

        count += 1

    # update the vocabulary with the combination one
    if detector_type == 'object_attr':
        vocabulary = obj_attr_vocabulary
        print('replace the vocabulary with object-attr combination')

    return roidb, len(vocabulary), vocabulary, image_list, tree_list


def clevr_obj_attr_vocab():
    objs = ['cube', 'cylinder', 'sphere']
    colors = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
    materials = ['metal', 'rubber']

    vocabulary = []
    for obj in objs:
        for color in colors:
            for mat in materials:
                word = obj + '-' + color + '-' + mat
                vocabulary.append(word)
    vocabulary = ['__background__'] + vocabulary

    vocabulary_dict = {
        'obj-attr-comb': vocabulary,
        'obj': objs,
        'color': colors,
        'material': materials
    }

    return vocabulary_dict


def assign_bbox_overlaps(overlaps, bbox):
    for i in range(0, overlaps.shape[0]):
        overlaps[i][int(bbox[i][-1])] = 1.0

    return overlaps


def get_vocabulary(tree_input, vocab_dir, read_tree=True):
    vocab_path = os.path.join(vocab_dir, 'vocabulary.pkl')
    if os.path.isfile(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        if read_tree:
            tree_list = os.listdir(tree_input)
            vocab = []
            for tree_file in tree_list:
                tree = pickle.load(open(osp.join(tree_input, tree_file), 'rb'))
                vocab = get_vocab(tree, vocab)
        else:
            vocab = []
            for tree in tree_input:
                vocab = get_vocab(tree, vocab)
        vocab = sorted(vocab)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=2)
    return vocab


def get_vocab(tree, vocab):
    if tree.word not in vocab:
        vocab.append(tree.word)

    for i in range(tree.num_children):
        vocab = get_vocab(tree.children[i], vocab)

    return vocab


def get_bbox_from_tree(tree_input, vocabulary, read_tree=True):
    if read_tree:
        tree = pickle.load(open(tree_input, 'rb'))
    else:
        tree = tree_input
    box_list = _get_bbox_from_tree(tree, vocabulary)
    return np.array(box_list, dtype=np.float32)


def _get_bbox_from_tree(tree, vocabulary):
    bbox_list = list()
    for i in range(tree.num_children):
        bbox_list += _get_bbox_from_tree(tree.children[i], vocabulary)

    if tree.function == 'describe':
        bbox_node = list(tree.bbox)
        bbox_node = [bbox_node[1], bbox_node[0], bbox_node[1] + bbox_node[3], bbox_node[0] + bbox_node[2]]
        bbox_node.append(vocabulary.index(tree.word))
        bbox_list.append(bbox_node)
        try:
            assert (len(bbox_list[-1]) == 5)
        except:
            print(bbox_list[-1], tree.word)

    return bbox_list


def get_objattr_from_tree(tree_input, vocabulary, read_tree=True):
    if read_tree:
        tree = pickle.load(open(tree_input, 'rb'))
    else:
        tree = tree_input

    objattr_list = _get_objattr_from_tree(tree, vocabulary, None)

    return objattr_list


def _get_objattr_from_tree(tree, vocabulary, properties):
    objattr_list = list()

    if tree.function == 'describe':
        properties = dict()
        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list
        properties['obj'] = tree.word
        try:
            label = properties['obj'] + '-' + properties['color'] + '-' + properties['material']
        except:
            print('properties {} is incomplete'.format(properties))
        vocab_idx = vocabulary['obj-attr-comb'].index(label)
        objattr_list.append(vocab_idx)

        return objattr_list, None

    elif tree.function == 'combine':
        if tree.word in vocabulary['color']:
            properties['color'] = tree.word
        elif tree.word in vocabulary['material']:
            properties['material'] = tree.word

        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list
        return objattr_list, properties

    else:  # layout module
        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list

        return objattr_list, None
