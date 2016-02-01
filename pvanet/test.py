import numpy as np
from skimage import io, transform
import os
import sys
import datetime
import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#prefix = '../data/pva/7.0.1/full/caffemodel.mx'
prefix = '../data/pva/7.0.1/full/PVANET7.0.1_pre1.7_anchor9_faster_rcnn_once_iter_1500000.caffemodel.mx'
num_round = 1
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

def PreprocessImage(path, size=(192, 192)):
    img = io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = transform.resize(crop_img, size)

    sample = np.asarray(resized_img) * 256
    if len(sample.shape) == 2:
        sample = np.tile(sample, (3, 1, 1))
    else:
        sample = sample[:,:,:3].swapaxes(0, 2).swapaxes(1, 2).copy()
    sample[0] -= 123.68
    sample[1] -= 116.779
    sample[2] -= 103.939

    sample = sample.reshape((1, sample.shape[0], sample.shape[1], sample.shape[2]))
    return sample

def Test():
    data = [line.strip().split(' ') for line in open('../meta/imagenet/val.txt', 'r').readlines()]
    img_paths = [line[0] for line in data]
    gt_labels = [int(line[1]) for line in data]
    acc_top1 = 0.0
    err_top5 = 0.0
    time = 0.0
    for i in range(0, 50000, 100):
        tps_top1, fns_top5, elapsed_time = Prediction(model, img_paths[i:i+100], gt_labels[i:i+100])
        acc_top1 = acc_top1 * (i / (i + 100.0)) + sum(tps_top1) / (i + 100.0)
        err_top5 = err_top5 * (i / (i + 100.0)) + sum(fns_top5) / (i + 100.0)
        time = time * (i / (i + 100.0)) + elapsed_time / (i + 100.0)
        print 'Top-1 Acc = {:.4f}, Top-5 Err = {:.4f}, Time = {:.2f}'.format(acc_top1, err_top5, time)

def Prediction(net, img_paths, gt_labels):
    batch = np.zeros((len(img_paths), 3, 192, 192))
    for i, img_path in enumerate(img_paths):
        batch[i] = PreprocessImage('../' + img_path)
    start_time = datetime.datetime.now()
    probs = model.predict(batch)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
    preds = [np.argsort(prob)[-5:][::-1] for prob in probs]
    tps_top1 = [int(gt) == pred[0] for pred, gt in zip(preds, gt_labels)]
    fns_top5 = [int(gt) not in pred for pred, gt in zip(preds, gt_labels)]
    return (tps_top1, fns_top5, elapsed_time)
