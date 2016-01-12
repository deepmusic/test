import numpy as np
from skimage import io, transform
from skimage.viewer import ImageViewer
import os
import sys
import datetime
import caffe
import config

caffe.set_mode_gpu()
#caffe.set_device(3)

dump_path = model_cfg['dump_path']
input_size = model_cfg['input_size']

mean_img = np.asarray(model_cfg['mean_img'], dtype=np.float32)
if len(mean_img.shape) == 1:
    if len(mean_img) == 1:
        mean_img = np.tile(mean_img, (3,))
    mean_img = mean_img[:3].reshape((3, 1, 1))
    mean_img = np.tile(mean_img, (1, input_size[0], input_size[1]))

model_synset_path = model_cfg['model_synset']
image_path = data_info['image_path']
gt_label_path = data_info['gt_label']
gt_synset_path = data_info['gt_synset']
blacklist_path = data_info['blacklist']

def PreprocessImage(path, input_size=input_size, mean_img=mean_img, show_img=False):
    img = io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = transform.resize(crop_img, input_size)

    if show_img:
        print('Original Image Shape: ', img.shape)
        viewer = ImageViewer(resized_img)
        viewer.show()

    sample = np.asarray(resized_img) * 256
    if len(sample.shape) == 2:
        sample = np.tile(sample, (3, 1, 1))
    else:
        sample = sample[:,:,:3][:,:,::-1]
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)

    sample -= mean_img
    sample = sample.reshape((1, 3, input_size[0], input_size[1])).copy()
    return sample

def Forward(net, sample, end=None):
    net.blobs['data'].reshape(*(sample.shape))
    forward_kwargs = {'data': sample.astype(np.float32, copy=False)}

    if end is not None:
        net.forward(end=end, **forward_kwargs)
    else:
        net.forward(**forward_kwargs)

def LoadNet(name, path=dump_path):
    name = os.path.join(path, name)
    net = caffe.Net(name + '.prototxt', name + '.caffemodel', caffe.TEST)
    return net

def Prediction(net, sample):
    net.blobs['data'].reshape(*(sample.shape))
    forward_kwargs = {'data': sample.astype(np.float32, copy=False)}

    start_time = datetime.datetime.now()
    blobs_out = net.forward(**forward_kwargs)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds() * 1000

    output = net.blobs['prob'].data
    top5 = [np.argsort(pred)[-5:][::-1] for pred in output]
    return (top5, elapsed_time)

def Performance(net, num_images):
    synset = [line.strip().split(' ')[0] for line in open(model_synset_path, 'r').readlines()]
    gt_labels = np.loadtxt(gt_label_path, dtype=np.int)
    gt_synset = [line.strip().split(' ')[1] for line in open(gt_synset_path).readlines()]
    gt_labels = [gt_synset[label-1] for label in gt_labels]
    if blacklist_path is not None:
        blacklist = np.loadtxt(blacklist_path, dtype=np.int)
    else:
        blacklist = []

    blacklist_count = 0
    data_count = 0.0
    err = 0.0
    time = 0.0
    batch_size = 10
    batch_count = 0
    samples = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    gt_samples = [None] * batch_size
    for idx in range(num_images):
        if blacklist_count < len(blacklist) and (idx+1) == blacklist[blacklist_count]:
            blacklist_count += 1
            continue

        sample = PreprocessImage(image_path % (idx+1))
        samples[batch_count] = sample[0]
        gt_samples[batch_count] = gt_labels[idx]

        batch_count += 1
        if batch_count < batch_size:
            continue
        batch_count = 0

        top5set, elapsed_time = Prediction(net, samples)
        top5set = [[synset[i] for i in top5] for top5 in top5set]

        data_count += batch_size
        err = err * (data_count - batch_size) / data_count \
                  + sum([gt_samples[i] not in top5 for i, top5 in enumerate(top5set)]) / data_count
        time = time * (data_count - batch_size) / data_count + elapsed_time / data_count

        print 'Image %d: Average Error=%.2f%%, Time=%.2fms' \
                % (idx, err*100, time)

if __name__ == "__main__":
    net = LoadNet(sys.argv[1])
    Performance(net, 50000)
