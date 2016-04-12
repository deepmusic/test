#!/bin/bash
import caffe
import numpy as np
import sys

sys.path.append('../test/compress')

import compress

net = caffe.Net('models/pva7.1.1_ori_bn/test_pvanet7.1.1_ori.prototxt', 'models/pva7.1.1_ori_bn/pvanet7.1.1_ori_bn_train_iter_1160000.caffemodel', caffe.TEST)

data = compress.PreprocessImage('../data/voc/2007/VOC2007/JPEGImages/000004.jpg', (192, 192), [103.939, 116.779, 123.68])
net.blobs['data'].reshape(*(data.shape))
net.forward(**{'data': data})

#data = compress.PreprocessImage('../data/voc/2007/VOC2007/JPEGImages/000004.jpg', (736, 576), [103.939, 116.779, 123.68])
#im_info = np.array([576, 736, 736/500.0, 576/406.0], dtype=np.float32).reshape(1, 4)
#net.blobs['data'].reshape(*(data.shape))
#net.forward(**{'data': data, 'im_info': im_info})
