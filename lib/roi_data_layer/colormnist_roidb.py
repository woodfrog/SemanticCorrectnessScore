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

""" COLORMNIST DATA PROCESSING"""


def prepare_colormnist_multi_data(directory, folder, phase, replace_image_dir=False, new_image_dir=None):
    treefile = osp.join(directory, folder, phase + '_parents.list')
    functionfile = osp.join(directory, folder, phase + '_functions.list')
    infofile = osp.join(directory, folder, phase + '_text.txt')
    # read info, trees, and functions
    with open(infofile, 'r') as f:
        info = []
        number = []
        for line in f:
            line = line[:-1].split(' ')
            number += [int(line[0][0])]
            line[0] = line[0][2:]
            info += [line]

    trees = read_trees(treefile)
    functions = read_functions(functionfile)

    image_list, trees, dictionary = parse_info_multi(directory, folder, phase, functions, trees, info, number)

    if replace_image_dir:
        print('replacing image dir with {}'.format(new_image_dir))
        assert new_image_dir is not None
        new_image_list = list()
        for i in range(len(trees)):
            new_image_list.append(osp.join(new_image_dir, 'image{:05d}.png'.format(i)))
        try:
            assert len(new_image_list) == len(image_list)
        except AssertionError:
            print('the new image list has different length of the standard image list, leading to potential unmatching')
        image_list = new_image_list

    return image_list, trees


def parse_info_multi(directory, folder, phase, functions, trees, info, number):
    imglist = []
    dictionary = []

    count = 0
    for tree in trees:
        imglist.append(osp.join(directory, folder, phase, 'image{:05d}.png'.format(count)))
        words, numbers, bboxes = _extract_info(info[count], number[count])
        dictionary = list(set(dictionary + words))
        trees[count] = _refine_tree_info(tree, words, functions[count], numbers, bboxes)
        count += 1

    return imglist, trees, dictionary


## Fixed size version

def prepare_colormnist_data(directory, folder, phase, replace_image_dir=False, new_image_dir=None):
    treefile = osp.join(directory, folder, phase + '_parents.list')
    functionfile = osp.join(directory, folder, phase + '_functions.list')
    infofile = osp.join(directory, folder, phase + '_text.txt')
    # read info, trees, and functions
    with open(infofile, 'r') as f:
        info = [line[:-1].split(' ') for line in f.readlines()]
    trees = read_trees(treefile)
    functions = read_functions(functionfile)

    if 'ONE' in folder:
        number = 1
    elif 'TWO' in folder:
        number = 2
    elif 'THREE' in folder:
        number = 3
    else:
        raise ValueError('Wrong sub folder.')

    image_list, trees, dictionary = parse_info(directory, folder, phase, functions, trees, info, number)

    if replace_image_dir:
        print('replacing image dir with {}'.format(new_image_dir))
        assert new_image_dir is not None
        new_image_list = list()
        for i in range(len(trees)):
            new_image_list.append(osp.join(new_image_dir, 'image{:05d}.png'.format(i)))
        try:
            assert len(new_image_list) == len(image_list)
        except AssertionError:
            print('the new image list has different length of the standard image list, leading to potential unmatching')
        image_list = new_image_list

    return image_list, trees


def read_trees(treefile):
    filename = treefile
    with open(filename, 'r') as f:
        trees = [read_tree(line) for line in f.readlines()]
    return trees


def read_tree(line):
    parents = map(int, line.split())
    trees = dict()
    root = None
    for i in xrange(1, len(parents) + 1):
        # if not trees[i-1] and parents[i-1]!=-1:
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                # if trees[parent-1] is not None:
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root


def read_functions(functionfile):
    f = open(functionfile)
    functions = []
    for line in f:
        functions += [line[:-1].split(" ")]
    return functions


def parse_info(directory, folder, phase, functions, trees, info, number):
    imglist = []
    dictionary = []

    count = 0
    for tree in trees:
        imglist.append(osp.join(directory, folder, phase, 'image{:05d}.png'.format(count)))
        words, numbers, bboxes = _extract_info(info[count], number)
        dictionary = list(set(dictionary + words))
        trees[count] = _refine_tree_info(tree, words, functions[count], numbers, bboxes)
        count += 1

    return imglist, trees, dictionary


def _extract_info(line, num):
    words = []
    numbers = []
    bboxes = []
    count = 0

    numid = 0
    for ele in line:
        words.append(ele)
        if _is_number(ele):
            numbers.append(ele)
            numid += 1
            if numid == num:
                count += 1
                break
        count += 1

    newid = count
    for i in range(0, newid):
        if _is_number(line[i]):
            bboxes += [[int(ele) for ele in line[count:count + 4]]]
            count += 4
        else:
            bboxes += [[]]

    return words, numbers, bboxes


def _is_number(n):
    try:
        int(n)
        return True
    except:
        return False


def _refine_tree_info(tree, words, functions, numbers, bboxes):
    for i in range(0, tree.num_children):
        tree.children[i] = _refine_tree_info(tree.children[i], words, functions, numbers, bboxes)

    tree.word = words[tree.idx]
    tree.function = functions[tree.idx]
    tree.bbox = np.array(bboxes[tree.idx])

    return tree


""" End COLORMNIST DATA PROCESSING"""


def prepare_colormnist_roidb(directory, folder, phase, detector_type, use_multi, replace_image_dir=False,
                             new_image_dir=None):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """

    roidb = []
    count = 0
    if not use_multi:
        image_list, tree_list = prepare_colormnist_data(directory, folder, phase, replace_image_dir=replace_image_dir,
                                                        new_image_dir=new_image_dir)
    else:
        image_list, tree_list = prepare_colormnist_multi_data(directory, folder, phase,
                                                              replace_image_dir=replace_image_dir,
                                                              new_image_dir=new_image_dir)

    base_dir = os.path.join(directory, folder)
    vocabulary = get_vocabulary(tree_list, vocab_dir=base_dir, read_tree=False)
    vocabulary = ['__background__'] + vocabulary

    if detector_type == 'object_attr':
        vocabulary_dict = colormnist_obj_attr_vocab()
        # need to change the vocabulary
        obj_attr_vocabulary = vocabulary_dict['obj-attr-comb']

    for im, tree in zip(image_list, tree_list):
        roidb.append({})
        roidb[count]['img_id'] = count
        roidb[count]['image'] = image_list[count]
        sizes = PIL.Image.open(image_list[count]).size
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
            obj_attr_gt_classes, _ = get_objattr_from_tree_cm(tree, vocabulary_dict, read_tree=False)
            roidb[count]['gt_classes'] = np.array(obj_attr_gt_classes)
        else:
            raise ValueError('Invalid detector type {}'.format(detector_type))
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


def get_objattr_from_tree_cm(tree_input, vocabulary, read_tree=True):
    if read_tree:
        tree = pickle.load(open(tree_input, 'rb'))
    else:
        tree = tree_input

    objattr_list = _get_objattr_from_tree_cm(tree, vocabulary, None)

    return objattr_list


def _get_objattr_from_tree_cm(tree, vocabulary, properties):
    objattr_list = list()

    if tree.function == 'describe':
        properties = dict()
        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree_cm(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list
        properties['obj'] = tree.word
        try:
            label = properties['obj'] + '-' + properties['color']
        except:
            print('properties {} is incomplete'.format(properties))
        vocab_idx = vocabulary['obj-attr-comb'].index(label)
        objattr_list.append(vocab_idx)

        return objattr_list, None

    elif tree.function == 'combine':
        if tree.word in vocabulary['color']:
            properties['color'] = tree.word

        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree_cm(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list
        return objattr_list, properties

    else:  # layout module
        for i in range(tree.num_children):
            child_objattr_list, properties = _get_objattr_from_tree_cm(tree.children[i], vocabulary, properties)
            objattr_list += child_objattr_list

        return objattr_list, None


def colormnist_obj_attr_vocab():
    objs = [str(i) for i in range(10)]
    colors = ['blue', 'cyan', 'green', 'magenta', 'red', 'yellow']

    vocabulary = []
    for obj in objs:
        for color in colors:
            word = obj + '-' + color
            vocabulary.append(word)
    vocabulary = ['__background__'] + vocabulary

    vocabulary_dict = {
        'obj-attr-comb': vocabulary,
        'obj': objs,
        'color': colors,
    }

    return vocabulary_dict


if __name__ == '__main__':
    image_list, trees = prepare_colormnist_data(directory='data/COLORMNIST', folder='TWO_5000_64_modified',
                                                phase='train')
    print(len(image_list))
    print(len(trees))
