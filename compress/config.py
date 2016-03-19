model_ = {
    'pvanet8.1.2': {
        'layers': [('conv1_1', 11), ('conv3_1', 48), \
                   ('conv3_2', 64), ('conv3_3', 48), ('conv4_1', 110), \
                   ('conv4_2', 110), ('conv4_3', 110), ('conv5_1', 220)], \
        'input_size': (224, 224), \
        'mean_img': [103.939, 116.779, 123.68], \
        'data': 'imagenet/train.txt', \
    }, \
    'vggnet': {
        'layers': [('conv1_2', 11), ('conv2_1', 25), ('conv2_2', 28), \
                   ('conv3_1', 52), ('conv3_2', 46), ('conv3_3', 56), \
                   ('conv4_1', 104), ('conv4_2', 92), ('conv4_3', 100), \
                   ('conv5_1', 232), ('conv5_2', 224), ('conv5_3', 214)], \
        'input_size': (224, 224), \
        'mean_img': [103.939, 116.779, 123.68], \
        'data': 'imagenet/train.txt', \
    }, \
}
