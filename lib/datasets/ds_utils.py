# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def get_gt_det_from_tree(tree, vocabulary):
    classes, bboxes = _get_gt_det_from_tree(tree, vocabulary)

    return classes, bboxes


def _get_gt_det_from_tree(tree, vocabulary):
    class_list = list()
    bbox_list = list()
    for i in range(tree.num_children):
        child_class_list, child_bbox_list = _get_gt_det_from_tree(tree.children[i], vocabulary)
        class_list += child_class_list
        bbox_list += child_bbox_list

    if tree.function == 'describe':
        bbox_node = list(tree.bbox)
        # notice: we need to adjust the bbox info here, in our data the order (y,x,h,w)
        bbox_node = [bbox_node[1], bbox_node[0], bbox_node[1] + bbox_node[3], bbox_node[0] + bbox_node[2]]
        bbox_list.append(bbox_node)
        class_list.append(vocabulary.index(tree.word))
        try:
            assert (len(bbox_list[-1]) == 4)
        except:
            print(bbox_list[-1], tree.word)

    return class_list, bbox_list
