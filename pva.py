model_cfg = {
    'layers': [('conv1_1', 32), ('conv1_2', 16), ('conv2_1', 8), ('conv2_2', 16), \
               ('conv3_1', 16), ('conv3_2', 32)], 
#, ('conv4_1', 32), ('conv4_2', 64), \
#               ('conv5_1', 64), ('conv5_2', 128), ('fc6', 512)], \
    'dump_path': './data/pva', \
    'input_size': (192, 192), \
    'mean_img': [103.939, 116.779, 123.68], \
    'model_synset': './Inception/synset.txt', \
}

