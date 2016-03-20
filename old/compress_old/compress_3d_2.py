import numpy as np
import datetime
from compress import MakeTrainingDataset, CompressConvLayer, CompressFCLayer

#############################################################################
# 5. Outer loop of the algorithm
#############################################################################

# 3-layer compression for a convolutional layer C' x C x kh x kw:
#   Rk x C x kh x 1,  R x Rk x 1 x kw,  C' x R x 1 x 1
#   In our experiments, we set Rk = R
def Run3D(cfgs):
    # Load original and compressed networks
    # Please prepare the following three files
    #   ./prototxt/compressed_3d.prototxt: compressed network prototxt
    true_net = caffe.Net('./prototxt/true.prototxt', \
                         './caffemodel/true.caffemodel', \
                         caffe.TEST)
    compressed_net = caffe.Net('./prototxt/compressed_3d.prototxt', \
                               './caffemodel/true.caffemodel', \
                               caffe.TEST)

    # Compress layer-by-layer, earlier layer first
    for i in range(len(cfgs['layers'])):
        layer, rank = cfgs['layers'][i]
        print '########### Compressing %s to rank %d ###########' \
              % (layer, rank)

        # Original network parameters
        W_true = true_net.params[layer][0].data
        b_true = true_net.params[layer][1].data

        # Compression
        if 'conv' in layer:
            # C' x C x kh x kw  ->  R x C x kh x 1  and  C' x R x 1 x kw
            W1, b1, W2, b2 = CompressConvKernel(W_true, b_true, rank)
            compressed_net.params[layer + '_kh'][0].data[:] = W1
            compressed_net.params[layer + '_kh'][1].data[:] = b1
            compressed_net.params[layer + '_kw'][0].data[:] = W2
            compressed_net.params[layer + '_kw'][1].data[:] = b2

            # C' x R x 1 x kw  ->  R x R x 1 x kw  and  C' x R x 1 x 1
            Y_true, Y_dirty = \
                MakeTrainingDataset(cfgs, layer, true_net, compressed_net,
                                    compressed_layer_postfix='_kw')
            W1, b1, W2, b2 = \
                CompressConvLayer(Y_true, Y_dirty, W2, b2, rank)
            compressed_net.params[layer + '_kw_a'][0].data[:] = W1
            compressed_net.params[layer + '_kw_a'][1].data[:] = b1
            compressed_net.params[layer + '_kw_b'][0].data[:] = W2
            compressed_net.params[layer + '_kw_b'][1].data[:] = b2
        elif 'fc' in layer:
            W1, b1, W2, b2 = CompressFCLayer(W_true, b_true, rank)
            compressed_net.params[layer + '_a'][0].data[:] = W1
            compressed_net.params[layer + '_a'][1].data[:] = b1
            compressed_net.params[layer + '_b'][0].data[:] = W2
            compressed_net.params[layer + '_b'][1].data[:] = b2

    # Save the compressed network as: ./caffemodel/compressed.caffemodel
    compressed_net.save('./caffemodel/compressed_3d.caffemodel')



#############################################################################
# 6. Main module
#############################################################################

import caffe

# model_: specification for compressed network & training data
#   layers: Layers to be compressed
#   data: Text file containing the list of paths to raw training images
#         Please prepare training image files and this image list file
#   input_size: Network input size
#   mean_bgr: Mean values of network input channels (in BGR order)
model_ = {
    'vggnet': {
        'layers': [('conv1_2', 11), ('conv2_1', 25), ('conv2_2', 28), \
                   ('conv3_1', 52), ('conv3_2', 46), ('conv3_3', 56), \
                   ('conv4_1', 104), ('conv4_2', 92), ('conv4_3', 100), \
                   ('conv5_1', 232), ('conv5_2', 224), ('conv5_3', 214), \
                   ('fc6', 512), ('fc7', 128)], \
        'data': './imagenet/train.txt', \
        'input_size': (224, 224), \
        'mean_bgr': [103.939, 116.779, 123.68], \
    }, \
}

if __name__ == "__main__":
    caffe.set_mode_gpu()
    caffe.set_device(0)

    Run3D(model_['vggnet'])
