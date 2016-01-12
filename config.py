model = {
    'pva': {
        'layers': [('conv1_1', 32), ('conv1_2', 16), ('conv2_1', 8), ('conv2_2', 16), \
                   ('conv3_1', 16), ('conv3_2', 32), ('conv4_1', 32), ('conv4_2', 64), \
                   ('conv5_1', 64), ('conv5_2', 128), ('fc6', 512)], \
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

data = {
    'imagenet': {
        'train': 'meta/imagenet/train.txt', \
        'test': 'meta/imagenet/test.txt', \
    }, \
