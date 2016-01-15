import numpy as np
from random import shuffle

model_ = {
    'pva': {
        'layers': [('conv1_2', 24), ('conv2_2', 48), \
                   ('conv3_2', 96), ('conv4_2', 192), \
                   ('conv5_2', 384)], \
        'input_size': (192, 192), \
        'mean_img': [103.939, 116.779, 123.68], \
    }, \
    'vgg': {
        'layers': [('conv1_1', 64), ('conv1_2', 11), ('conv2_1', 25), ('conv2_2', 28), \
                   ('conv3_1', 52), ('conv3_2', 46), ('conv3_3', 56), \
                   ('conv4_1', 104), ('conv4_2', 92), ('conv4_3', 100), \
                   ('conv5_1', 232), ('conv5_2', 224), ('conv5_3', 214), \
                   ('fc6', 512), ('fc7', 1024), ('fc8', 100)], \
        'input_size': (224, 224), \
        'mean_img': [103.939, 116.779, 123.68], \
    }, \
}

data_ = {
    'imagenet': {
        'train': 'meta/imagenet/train.txt', \
        'test': 'meta/imagenet/val.txt', \
        'image_size': (256, 256), \
    }, \
    'voc2007': {
        'train': 'meta/voc2007/data.txt', \
        'test': 'meta/voc2007/data.txt', \
        'image_size': (256, 256), \
    }
}

class Config(object):
    def __init__(self, model_name, version, data_name):
        base_path = model_name + '/' + version + '/low'
        self.meta_path = 'meta/' + base_path
        self.data_path = 'data/' + base_path
        self.dump_path = self.data_path + '/dump'

        data = data_[data_name]
        self.train = data['train']
        self.test = data['test']
        self.image_size = data['image_size']

        model = model_[model_name]
        self.layers = model['layers']
        self.input_size = model['input_size']
        self.mean_img = model['mean_img']
        self.Mean2Tensor()

    def Mean2Tensor(self):
        mean_img = np.asarray(self.mean_img, dtype=np.float32)
        if len(mean_img.shape) == 1:
            if len(mean_img) == 1:
                mean_img = np.tile(mean_img, (3,))
            mean_img = mean_img[:3].reshape((3, 1, 1))
            mean_img = np.tile(mean_img, (1, self.input_size[0], self.input_size[1]))
        self.mean_img = mean_img

    def TrainImageNames(self, do_shuffle=True):
        image_names = [line.strip().split(' ')[0] for line in open(self.train, 'r').readlines()]
        if do_shuffle:
            shuffle(image_names)
        return image_names

    def TestImageNames(self, do_shuffle=True):
        image_names = [line.strip().split(' ')[0] for line in open(self.test, 'r').readlines()]
        if do_shuffle:
            shuffle(image_names)
        return image_names

    def TrainImageNameLabelPairs(self, do_shuffle=True):
        pairs = [line.strip().split(' ') for line in open(self.train, 'r').readlines()]
        pairs = [(pair[0], int(pair[1])) for pair in pairs]
        if do_shuffle:
            shuffle(pairs)
        return pairs

    def TestImageNameLabelPairs(self, do_shuffle=True):
        pairs = [line.strip().split(' ') for line in open(self.test, 'r').readlines()]
        pairs = [(pair[0], int(pair[1])) for pair in pairs]
        if do_shuffle:
            shuffle(pairs)
        return pairs
