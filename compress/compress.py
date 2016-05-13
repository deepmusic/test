#############################################################################
# Contents
#   1. Global hyper-parameters
#   2. Training data collector modules
#   3. Convolutional layer compression modules
#   4. Convolutional kernel compression modules (for 3-layer compression)
#   5. Fully-connected layer compression modoules
#   6. Outer loop of the algorithm
#   7. Main module
#############################################################################

import numpy as np
import skimage
import datetime
from random import shuffle



#############################################################################
# 1. Global hyper-parameters
#############################################################################

# Total number of training data = num_train_images * sample_per_image = 48000
#   num_train_images: Number of training images... 500 is enough for VGG
#   sample_per_image: Number of output patches (channels x 1 x 1)
#                     randomly sampled for each image
num_train_images = 2000
sample_per_image = 48



#############################################################################
# 2. Training data collector modules
#############################################################################

# Load a training image, and then transform it into a network input
def PreprocessImage(path, size, mean_bgr):
    img = skimage.io.imread(path)

    # Center square crop and then resize to the network input size
    short_edge = min(img.shape[:2])
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    cropped = img[y:y+short_edge, x:x+short_edge]
    resized = skimage.transform.resize(cropped, size)

    # Scaling value range:  [0, 1]  ->  [0, 256]
    # Reordering:  height x width x channels  ->  channels x height x width
    # Channel swap:  RGB  ->  BGR
    sample = np.array(resized) * 256
    if len(sample.shape) == 2:
        sample = np.tile(sample, (3, 1, 1))
    else:
        sample = sample[:,:,:3][:,:,::-1].swapaxes(0, 2).swapaxes(1, 2)

    # Subtract the mean value of each channel
    sample[0] -= mean_bgr[0]
    sample[1] -= mean_bgr[1]
    sample[2] -= mean_bgr[2]

    # Make a 4-dimensional tensor:  1 x channels x height x width  (NCHW)
    sample = np.tile(sample, (1, 1, 1, 1))

    return sample


# For a given network input, forward-pass until getting given layer's output
def Forward(net, input_data, output_layer=None):
    net.blobs['data'].reshape(*(input_data.shape))
    if output_layer is not None:
        net.forward(end=output_layer, **{'data': input_data})
    else:
        net.forward(**{'data': input_data})


# Randomly choose patches (channels x 1 x 1)
# from a given layer (images x channels x width x height),
# then each patch will be a training data point
def ParseOutputLayer(y_4d, subset=None):
    if len(y_4d.shape) == 2:
        if subset is None:
            return (y_4d, [])
        return y_4d

    # Skip boundary patches that come from lots of zero-padded input data
    num_data = y_4d.shape[0] * (y_4d.shape[2]-2) * (y_4d.shape[3]-2)
    ndim = y_4d.shape[1]
    y_4d_nhwc = y_4d.swapaxes(1, 3).swapaxes(1, 2)
    y_2d = y_4d_nhwc[:, 1:y_4d.shape[2]-1, 1:y_4d.shape[3]-1, :] \
           .reshape((num_data, ndim))

    # Randomly sample in one network (e.g., original net),
    # and then sample from the same positions in another network
    # (e.g., compressed net)
    if subset is None:
        subset = np.random.permutation(num_data)[:sample_per_image]
        return (y_2d[subset], subset)
    return y_2d[subset]


# Get training data for a layer from an image file
#   y_true:  outputs randomly-sampled from a layer in the original net
#   y_dirty:  outputs (at the same positions) in the compressed net
#             inputs are come from the previous layer's outputs
#             the outputs are not compressed yet, but 'dirty'
#             due to compression errors accumulated from previous layers
def MakeTrainingDataSub(image_name, size, mean_bgr, \
                        layer, true_net, compressed_net, **kwargs):
    sample = PreprocessImage(image_name, size, mean_bgr)

    # If the target layer has different names in two networks,
    # append appropriate postfix to each network
    true_layer = layer
    if kwargs.has_key('true_layer_postfix'):
        true_layer += kwargs['true_layer_postfix']
    compressed_layer = layer
    if kwargs.has_key('compressed_layer_postfix'):
        compressed_layer += kwargs['compressed_layer_postfix']

    # Do forward-passing until obtaining the outputs of the given layer
    Forward(true_net, sample, true_layer)
    Forward(compressed_net, sample, compressed_layer)

    # Randomly sample y_true in the original net,
    # and then sample y_dirty from the same positions in the compressed net
    y_true, subset = ParseOutputLayer(true_net.blobs[true_layer].data)
    y_dirty = ParseOutputLayer(compressed_net.blobs[compressed_layer].data, \
                               subset)

    return (y_true, y_dirty)


# Outer module for collecting training data
def MakeTrainingDataset(cfgs, layer, true_net, compressed_net, **kwargs):
    # Load the list of training images
    image_names = [line.strip().split(' ')[0] \
                   for line in open(cfgs['data'], 'r').readlines()]
    shuffle(image_names)
    image_names = image_names[:num_train_images]

    # size: Network input size
    # mean_bgr: Mean values of network input channels (in BGR order)
    size = cfgs['input_size']
    mean_bgr = cfgs['mean_bgr']

    # Get training data from the first image in the list,
    # and then determine the dimension of training data points
    y_true, y_dirty = MakeTrainingDataSub(image_names[0], size, mean_bgr, \
                                          layer, true_net, compressed_net, \
                                          **kwargs)
    Y_true = np.zeros((len(image_names), y_true.shape[0], y_true.shape[1]), \
                      dtype=np.float32)
    Y_dirty = np.zeros((len(image_names), y_true.shape[0], y_true.shape[1]), \
                       dtype=np.float32)
    Y_true[0] = y_true
    Y_dirty[0] = y_dirty

    # Collect training data from the remaining images in the list
    start_time = datetime.datetime.now()
    for i in range(1, len(image_names)):
        Y_true[i], Y_dirty[i] = \
            MakeTrainingDataSub(image_names[i], size, mean_bgr, \
                                layer, true_net, compressed_net)

        # Since it takes somewhat long time, print status periodically
        if (i+1) % 10 == 0:
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print '[%6d image processed] Time=%.2fs' % (i+1, elapsed_time)
            start_time = end_time

    # For accurate compression, store data in double-precision format
    Y_true = Y_true.reshape((np.prod(Y_true.shape[:2]), Y_true.shape[2])) \
                   .swapaxes(0, 1).astype(np.float64, copy=True)
    Y_dirty = Y_dirty.reshape((np.prod(Y_dirty.shape[:2]), Y_dirty.shape[2])) \
                     .swapaxes(0, 1).astype(np.float64, copy=True)

    return (Y_true, Y_dirty)



#############################################################################
# 3. Convolutional layer compression modules
#############################################################################

# Measure compression error and accuracy
#   error  =  L2Loss[ ReLU(Y_true)  -  ReLU(M * Y_dirty + b) ]
#   accuracy  =  CosineSimilarity[ ReLU(Y_true),  ReLU(M * Y_dirty + b) ]
def ConvCompressionError(Y_true, Y_dirty, M, b):
    # Y_nonlinear  =  ReLU(Y_true)
    Y_nonlinear = Y_true.copy()
    Y_nonlinear[Y_nonlinear < 0] = 0

    # Z_nonlinear  =  ReLU(M * Y_dirty + b)
    Z_nonlinear = np.dot(M, Y_dirty)
    for i in range(Z_nonlinear.shape[0]):
        Z_nonlinear[i] += b[i]
    Z_nonlinear[Z_nonlinear < 0] = 0

    # Compute error and accuracy
    err = ((Y_nonlinear - Z_nonlinear) ** 2).sum()
    yTz = np.multiply(Y_nonlinear, Z_nonlinear).sum(axis=0)
    y_norm = np.sqrt((Y_nonlinear ** 2).sum(axis=0))
    z_norm = np.sqrt((Z_nonlinear ** 2).sum(axis=0))
    cossim = np.divide(yTz, 1e-10 + np.multiply(y_norm, z_norm)).mean()

    return (err, cossim)


# Optimization with respect to the auxiliary variable z
#   min_Z  L2Loss[ ReLU(Y_true) -  ReLU(Z) ]
#          +  penalty * L2Loss[ Z - (M * Y_dirty + b) ]
def CompressConvLayerAuxiliary(Y_true, Y_dirty, M, b, penalty):
    # Y_nonlinear  =  ReLU(Y_true)
    Y_nonlinear = Y_true.reshape((np.prod(Y_true.shape),)).copy()
    Y_nonlinear[Y_nonlinear < 0] = 0

    # Y_approx  =  M * Y_dirty + b
    Y_approx = np.dot(M, Y_dirty)
    for i in range(Y_true.shape[0]):
        Y_approx[i] += b[i]
    Y_approx = Y_approx.reshape((np.prod(Y_true.shape),))

    # Compute the closed form solution for z
    Z_minus = Y_approx.copy()
    Z_minus[Z_minus > 0] = 0
    Z_plus = (penalty * Y_approx + Y_nonlinear) / (penalty + 1)
    Z_plus[Z_plus < 0] = 0
    err_minus = (Y_nonlinear ** 2) \
                   + penalty * ((Z_minus - Y_approx) ** 2)
    err_plus = ((Y_nonlinear - Z_plus) ** 2) \
                   + penalty * ((Z_plus - Y_approx) ** 2)
    minus_is_better = err_minus < err_plus
    Z_plus[minus_is_better] = Z_minus[minus_is_better]
    Z_optimal = Z_plus.reshape((Y_true.shape[0], Y_true.shape[1]))

    return Z_optimal


# Core module for convolutional layer compression
#   Alternating optimization:
#   1.  min_Z  L2Loss[ ReLU(Y_true) -  ReLU(Z) ]
#              +  penalty * L2Loss[ Z - (M * Y_dirty + b) ]
#   2.  min_{M, b}  L2Loss[ Z  -  (M * Y_dirty + b) ]
def CompressConvLayerCore(Y_true, Y_dirty, rank,
                          penalty=None, M=None, b=None):
    start_time = datetime.datetime.now()

    # Optimization with respect to the auxiliary variable z
    if penalty is None:
        Z = Y_true.copy()
    else:
        Z = CompressConvLayerAuxiliary(Y_true, Y_dirty, M, b, penalty)

    # Centering Z and Y_dirty
    mu_Z = Z.mean(axis=1)
    for i in range(Z.shape[0]):
        Z[i] -= mu_Z[i]
    Y = Y_dirty.copy()
    mu_Y = Y.mean(axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= mu_Y[i]

    # Optimization with respect to M
    #   min_M  L2Loss[ (Z - mean(Z))  -  M * (Y_dirty - mean(Y_dirty)) ]
    ZY = np.tensordot(Z, Y, (1, 1))
    M_new = np.dot(ZY, np.linalg.inv(np.tensordot(Y, Y, (1, 1))))

    # Decompose M ~= P * Q (= Truncased SVD)
    D, U = np.linalg.eigh(np.tensordot(ZY, M_new, (1, 1)))
    S = np.sqrt(D[-rank:])
    U = U[:, -rank:]
    SinvUT = np.tensordot(np.diag(1.0 / S), U, (1, 1))
    VT = np.dot(SinvUT, M_new)
    Ssqrt = np.diag(np.sqrt(S))
    P = np.dot(U, Ssqrt)
    Q = np.dot(Ssqrt, VT)
    M_new = np.dot(P, Q)

    # Optimization with respect to b
    #   L2Loss[ (Z - mean(Z))  -  M * (Y_dirty - mean(Y_dirty)) ]
    #   =  L2Loss[ Z  -  M * Y_dirty  -  (mean(Z) - M * mean(Y_dirty)) ]
    #   =  L2Loss[ Z  -  (M * Y_dirty + b) ]
    # Therefore,
    #   b = mean(Z) - M * mean(Y_dirty)
    b_new = mu_Z - np.dot(M_new, mu_Y)

    # Print compression error, accuracy, and time
    err, cossim = ConvCompressionError(Y_true, Y_dirty, M_new, b_new)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Error = %.4f, Accuracy = %.4f, Elapsed time = %.2f' \
          % (err, cossim, elapsed_time)

    return (M_new, b_new, P, Q)


# Outer module for convolutional layer compression
def CompressConvLayer(Y_true, Y_dirty, W_true, b_true, rank):
    # Iterative optimization with more and more considering nonlinearity
    #   ReLU(Y_true) ~= ReLU(M * Y_dirty + b) = ReLU(P * Q * Y_dirty + b)
    M, b, P, Q = CompressConvLayerCore(Y_true, Y_dirty, rank)
    for i in range(25):
        M, b, P, Q = CompressConvLayerCore(Y_true, Y_dirty, rank, 0.01, M, b)
    for i in range(25):
        M, b, P, Q = CompressConvLayerCore(Y_true, Y_dirty, rank, 1.0, M, b)

    # Reshape W_true:  C' x C x kh x kw  ->  C' x (C * kh * kw)
    W_2d = W_true.reshape((W_true.shape[0], np.prod(W_true.shape[1:])))
    b_1d = b_true.reshape((b_true.shape[0],))

    # Parameters for the compression layer
    # y  ~=  P * Q * y + b  =  P * Q * (W_true * x + b_true) + b
    #                       =  P * (Q * W_true * x + Q * b_true) + b
    #                       =  W2 * (W1 * x + b1) + b2
    # Therefore,
    #   W1  =  Q * W_true,  b1  =  Q * b_true,  W2  =  P,  b2  =  b
    W1 = np.dot(Q, W_2d).astype(np.float32, copy=True)
    b1 = np.dot(Q, b_1d).astype(np.float32, copy=True)
    W2 = P.astype(np.float32, copy=True)
    b2 = b.astype(np.float32, copy=True)

    # Reshape W1:  R x (C * kh * kw)  ->  R x C x kh x kw
    #         W2:  C' x C  ->  C' x C x 1 x 1
    if len(W_true.shape) == 4:
        W1 = W1.reshape((Q.shape[0], \
                         W_true.shape[1], W_true.shape[2], W_true.shape[3]))
        W2 = W2.reshape((P.shape[0], P.shape[1], 1, 1))

    return (W1, b1, W2, b2)



#############################################################################
# 4. Convolutional kernel compression modules
#############################################################################

# Core module for convolutional kernel compression
#   Decompose convolutional kernel:  kh x kw  ->  kh x 1  and  1 x kw
#   Thus,  C' x C x kh x kw  ->  Rk x C x kh x 1  and  C' x Rk x 1 x kw
#   This can be used for 3-layer compression:
#     Rk x C x kh x 1,  R x Rk x 1 x kw,  C' x R x 1 x 1
def CompressConvKernelCore(W, rank, H=None, G=None):
    start_time = datetime.datetime.now()

    if H is None:
        H = np.random.normal(size=(W.shape[1]*W.shape[2], rank))

    W_G = W.swapaxes(2, 3).swapaxes(1, 2) \
           .reshape((W.shape[0]*W.shape[3], W.shape[1]*W.shape[2]))
    W_H = W.swapaxes(0, 1).swapaxes(1, 2) \
           .reshape((W.shape[1]*W.shape[2], W.shape[0]*W.shape[3]))
    G = np.dot(W_G, np.dot(H, np.linalg.inv(np.tensordot(H, H, (0, 0)))))
    H = np.dot(W_H, np.dot(G, np.linalg.inv(np.tensordot(G, G, (0, 0)))))

    # Print compression error and running time
    err = ((W_G - np.tensordot(G, H, (1, 1))) ** 2).sum()
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Kernel approximation error = %.4f, Elapsed time = %.2f' \
          % (err, elapsed_time)

    return (H, G)

# Outer module for convolutional kernel compression
def CompressConvKernel(W_true, b_true, rank):
    H, G = CompressConvKernelCore(W_true, rank)
    for i in range(1, 30):
        H, G = CompressConvKernelCore(W_true, rank, H, G)

    W1 = H.reshape((1, W_true.shape[1], W_true.shape[2], rank)) \
          .swapaxes(0, 3).astype(np.float32, copy=True)
    b1 = np.zeros((rank,), dtype=np.float32)
    W2 = G.reshape((W_true.shape[0], W_true.shape[3], 1, rank)) \
          .swapaxes(1, 3).astype(np.float32, copy=True)
    b2 = b_true.copy()

    return (W1, b1, W2, b2)



#############################################################################
# 5. Fully-connected layer compression modoules
#############################################################################

# Core module for FC layer compression (= Truncaated SVD)
def CompressFCLayerCore(W_true, rank):
    start_time = datetime.datetime.now()

    if W_true.shape[0] > W_true.shape[1]:
        W = W_true.transpose().copy()
    else:
        W = W_true.copy()

    # Do eigen decomposition along smaller dimension
    WWT = np.tensordot(W, W, (1, 1))
    D, U = np.linalg.eigh(WWT)
    U = U[:, -rank:]
    S_inv = np.diag(1.0 / np.sqrt(D[-rank:]))
    VT = np.dot(np.tensordot(S_inv, U, (1, 1)), W)
    S_sqrt = np.diag(np.sqrt(np.sqrt(D[-rank:])))

    # Obtain P, Q such that W_true ~= P * Q
    if W_true.shape[0] > W_true.shape[1]:
        P = np.dot(VT.transpose(), S_sqrt)
        Q = np.dot(S_sqrt, U.transpose())
    else:
        P = np.dot(U, S_sqrt)
        Q = np.dot(S_sqrt, VT)
    energy = D[-rank:].sum() / D.sum()

    # Print approximation accuracy in terms of L2 loss
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Truncated SVD Accumulative Energy = %.4f' % energy
    return (P, Q)


# Outer module for FC layer compression
def CompressFCLayer(W_true, b_true, rank):
    # Decompose original weight matrix:  W_true ~= P * Q
    #   W_true:  D' x D,  P:  D' x R,  Q:  R x D
    P, Q = CompressFCLayerCore(W_true, rank)

    # Parameters for the compression layer
    #   Compression layer (R x D):  W1 = Q,  b1 = 0
    #   Reconstruction layer (D' x R):  W2 = P,  b2 = b_true
    W1 = Q.astype(np.float32, copy=True)
    b1 = np.zeros((Q.shape[0],), dtype=np.float32)
    W2 = P.astype(np.float32, copy=True)
    b2 = b_true.copy()

    return (W1, b1, W2, b2)



#############################################################################
# 6. Outer loop of the algorithm
#############################################################################

def Compression(cfgs):
    # Load original and compressed networks
    # Please prepare the following three files
    #   ./prototxt/true.prototxt: original network prototxt
    #   ./prototxt/compressed.prototxt: compressed network prototxt
    #   ./caffemodel/true.caffemodel: original pre-trained caffemodel
    true_net = caffe.Net('./prototxt/true.prototxt', \
                         './caffemodel/true.caffemodel', \
                         caffe.TEST)
    compressed_net = caffe.Net('./prototxt/compressed.prototxt', \
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
            Y_true, Y_dirty = \
                MakeTrainingDataset(cfgs, layer, true_net, compressed_net)
            W1, b1, W2, b2 = \
                CompressConvLayer(Y_true, Y_dirty, W_true, b_true, rank)
        elif 'fc' in layer:
            W1, b1, W2, b2 = CompressFCLayer(W_true, b_true, rank)

        # Update compressed network parameters
        compressed_net.params[layer + '_a'][0].data[:] = W1
        compressed_net.params[layer + '_a'][1].data[:] = b1
        compressed_net.params[layer + '_b'][0].data[:] = W2
        compressed_net.params[layer + '_b'][1].data[:] = b2

    # Save the compressed network as: ./caffemodel/compressed.caffemodel
    compressed_net.save('./caffemodel/compressed.caffemodel')


# 3-layer compression for a convolutional layer C' x C x kh x kw:
#   Rk x C x kh x 1,  R x Rk x 1 x kw,  C' x R x 1 x 1
#   In our experiments, we set Rk = R
def Compression3D(cfgs):
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
        print '########### 3D-Compressing %s to rank %d ###########' \
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
                MakeTrainingDataset(cfgs, layer, true_net, compressed_net, \
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

    # Save the compressed network as: ./caffemodel/compressed_3d.caffemodel
    compressed_net.save('./caffemodel/compressed_3d.caffemodel')



#############################################################################
# 7. Main module
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
                   ('conv5_1', 232), ('conv5_2', 224), ('conv5_3', 214)], \
                   #('fc6', 512), ('fc7', 128)], \
        'data': './imagenet/train.txt', \
        'input_size': (224, 224), \
        'mean_bgr': [103.939, 116.779, 123.68], \
    }, \
    'vggnet3d': {
        'layers': [('conv1_2', 20), ('conv2_1', 44), ('conv2_2', 49), \
                   ('conv3_1', 91), ('conv3_2', 81), ('conv3_3', 98), \
                   ('conv4_1', 182), ('conv4_2', 161), ('conv4_3', 175), \
                   ('conv5_1', 406), ('conv5_2', 392), ('conv5_3', 375), \
                   ('fc6', 512), ('fc7', 128)], \
        'data': './imagenet/train.txt', \
        'input_size': (224, 224), \
        'mean_bgr': [103.939, 116.779, 123.68], \
    }, \
}

if __name__ == "__main__":
    caffe.set_mode_gpu()
    caffe.set_device(1)

    Compression(model_['vggnet'])
