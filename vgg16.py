model_cfg = {
    'layers': [('conv1_1', 64), ('conv1_2', 11), ('conv2_1', 25), ('conv2_2', 28), \
               ('conv3_1', 52), ('conv3_2', 46), ('conv3_3', 56), \
               ('conv4_1', 104), ('conv4_2', 92), ('conv4_3', 100), \
               ('conv5_1', 232), ('conv5_2', 224), ('conv5_3', 214), \
               ('fc6', 512), ('fc7', 1024), ('fc8', 100)], \
    'dump_path': './data/vgg16/models', \
    'input_size': (224, 224), \
    'mean_img': [103.939, 116.779, 123.68], \
    'model_synset': './Inception/synset.txt', \
}

