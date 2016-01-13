import numpy as np
from skimage import io, transform
from skimage.viewer import ImageViewer
import os
import sys
import datetime
import caffe
from config import Config

caffe.set_mode_gpu()
#caffe.set_device(3)

def PreprocessImage(path, mean_img, show_img=False):
    img = io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = transform.resize(crop_img, mean_img.shape[1:])

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
    sample = sample.reshape((1, sample.shape[0], sample.shape[1], sample.shape[2])).copy()
    return sample

def Forward(net, sample, end=None):
    net.blobs['data'].reshape(*(sample.shape))
    forward_kwargs = {'data': sample.astype(np.float32, copy=False)}

    if end is not None:
        net.forward(end=end, **forward_kwargs)
    else:
        net.forward(**forward_kwargs)

def LoadNet(name, config):
    prototxt = os.path.join(config.meta_path, name + '.prototxt')
    caffemodel = os.path.join(config.data_path, name + '.caffemodel')
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
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

def Performance(net, config, num_images):
    test_pairs = config.TestImageNameLabelPairs()
    mean_img = config.mean_img

    data_count = 0.0
    err = 0.0
    time = 0.0
    batch_size = 10
    batch_count = 0
    samples = np.zeros((batch_size, 3, mean_img.shape[1], mean_img.shape[2]), dtype=np.float32)
    gt_samples = [None] * batch_size

    for idx, (image_name, gt_label) in enumerate(test_pairs):
        sample = PreprocessImage(image_name, mean_img)
        samples[batch_count] = sample[0]
        gt_samples[batch_count] = gt_label

        batch_count += 1
        if batch_count < batch_size:
            continue
        batch_count = 0

        top5set, elapsed_time = Prediction(net, samples)

        data_count += batch_size
        err = err * (data_count - batch_size) / data_count \
                  + sum([gt_samples[i] not in top5 for i, top5 in enumerate(top5set)]) / data_count
        time = time * (data_count - batch_size) / data_count + elapsed_time / data_count

        if (idx+1) % 1000 == 0:
            print '[%6d image processed] Average Error=%.2f%%, Average Time=%.2fms' \
                    % (idx+1, err*100, time)
    print 'Average Error=%.2f%%, Average Time=%.2fms' % (err*100, time)

if __name__ == "__main__":
    model_name = 'pva'
    version = '7.0.1'
    data_name = 'imagenet'
    config = Config(model_name, version, data_name)

    net = LoadNet(sys.argv[1], config)
    Performance(net, config, 50000)
