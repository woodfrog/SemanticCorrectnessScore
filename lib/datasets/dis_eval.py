# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def dis_detector_eval(all_boxes,
                      roidb,
                      vocabulary,
                      selected_class_idx,
                      ovthresh=0.5,
                      use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the evaluation for detector.

    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    # ground-truth bboxes for every class in each image
    npos_all = [0 for _ in range(len(vocabulary))]
    gt_records = [[{'bbox': list(), 'difficult': list(), 'det': list()} for _ in range(len(vocabulary))] for _ in
                  range(len(roidb))]
    for image_id, roi_info in enumerate(roidb):
        gt_classes = roi_info['gt_classes'].tolist()
        gt_bboxes = roi_info['boxes'].tolist()
        assert (len(gt_classes) == len(gt_bboxes))
        for gt_class, gt_bbox in zip(gt_classes, gt_bboxes):
            gt_class = int(gt_class)
            gt_records[image_id][gt_class]['bbox'].append(gt_bbox)
            gt_records[image_id][gt_class]['difficult'].append(0)
            gt_records[image_id][gt_class]['det'].append(False)
            npos_all[gt_class] = npos_all[gt_class] + 1

    # one entry for each class
    image_ids_all = [[] for _ in range(len(vocabulary))]
    confidence_all = [[] for _ in range(len(vocabulary))]
    BB_all = [[] for _ in range(len(vocabulary))]

    for class_idx in range(len(vocabulary)):
        for image_id in range(len(all_boxes[class_idx])):
            # iterate through each det
            for det in all_boxes[class_idx][image_id]:
                image_ids_all[class_idx].append(image_id)
                confidence_all[class_idx].append(det[4])
                BB_all[class_idx].append(list(det[:4]))
    aps = list()
    # compute map for every selected class
    for class_idx in selected_class_idx:
        npos = npos_all[class_idx]

        image_ids = image_ids_all[class_idx]
        confidence = np.array(confidence_all[class_idx])
        BB = np.array(BB_all[class_idx])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = gt_records[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = np.array(R[class_idx]['bbox']).astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R[class_idx]['difficult'][jmax]:
                        if not R[class_idx]['det'][jmax]:
                            tp[d] = 1.
                            R[class_idx]['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        print('rec: ', rec)
        print('prec: ', prec)
        print('ap: ', ap)
        if not np.isnan(ap):
            aps.append(ap)

    aps = np.array(aps)

    return aps.mean()


def detector_score_eval(all_boxes,
                        roidb,
                        vocabulary,
                        selected_class_idx,
                        use_07_metric=False):
    # ground-truth bboxes for every class in each image
    npos_all = [0 for _ in range(len(vocabulary))]
    gt_records = [[{'bbox': list(), 'difficult': list(), 'det': list()} for _ in range(len(vocabulary))] for _ in
                  range(len(roidb))]
    for image_id, roi_info in enumerate(roidb):
        gt_classes = roi_info['gt_classes'].tolist()
        gt_bboxes = roi_info['boxes'].tolist()
        assert (len(gt_classes) == len(gt_bboxes))
        for gt_class, gt_bbox in zip(gt_classes, gt_bboxes):
            gt_class = int(gt_class)
            gt_records[image_id][gt_class]['bbox'].append(gt_bbox)
            gt_records[image_id][gt_class]['difficult'].append(0)
            gt_records[image_id][gt_class]['det'].append(False)
            npos_all[gt_class] = npos_all[gt_class] + 1

    # one entry for each class
    image_ids_all = [[] for _ in range(len(vocabulary))]
    confidence_all = [[] for _ in range(len(vocabulary))]
    BB_all = [[] for _ in range(len(vocabulary))]

    for class_idx in range(len(vocabulary)):
        for image_id in range(len(all_boxes[class_idx])):
            # iterate through each det
            for det in all_boxes[class_idx][image_id]:
                image_ids_all[class_idx].append(image_id)
                confidence_all[class_idx].append(det[4])
                BB_all[class_idx].append(list(det[:4]))

    aps = list()
    # compute map for every selected class
    for class_idx in selected_class_idx:
        npos = npos_all[class_idx]

        image_ids = image_ids_all[class_idx]
        confidence = np.array(confidence_all[class_idx])
        BB = np.array(BB_all[class_idx])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = gt_records[image_ids[d]]
                bb = BB[d, :].astype(float)
                BBGT = np.array(R[class_idx]['bbox']).astype(float)

                if BBGT.size > 0:
                    matched_idx = 0
                    while matched_idx < len(R[class_idx]['det']) and R[class_idx]['det'][matched_idx] is True:
                        matched_idx += 1

                    if matched_idx < len(R[class_idx]['det']):
                        tp[d] = 1.
                        R[class_idx]['det'][matched_idx] = True
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        if npos == 0:
            rec = tp
        else:
            rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        if npos > 0:
            print('fp: ', fp)
            print('rec: ', rec)
            print('prec: ', prec)
            print('ap: ', ap)
            aps.append(ap)

    aps = np.array(aps)

    return aps.mean()
