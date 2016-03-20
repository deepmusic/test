import numpy as np
import skimage
import datetime
from random import shuffle

num_train_images = 500
sample_per_image = 96

def PreprocessImage(path, size, mean_bgr):
    img = skimage.io.imread(path)

    short_edge = min(img.shape[:2])
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    cropped = img[y:y+short_edge, x:x+short_edge]
    resized = skimage.transform.resize(cropped, size)

    sample = np.array(resized) * 256
    if len(sample.shape) == 2:
        sample = np.tile(sample, (3, 1, 1))
    else:
        sample = sample[:,:,:3][:,:,::-1].swapaxes(0, 2).swapaxes(1, 2)

    sample[0] -= mean_bgr[0]
    sample[1] -= mean_bgr[1]
    sample[2] -= mean_bgr[2]
    sample = np.tile(sample, (1, 1, 1, 1))

    return sample

def Forward(net, input_data, output_layer=None):
    net.blobs['data'].reshape(*(input_data.shape))
    if output_layer is not None:
        net.forward(end=output_layer, **{'data': input_data})
    else:
        net.forward(**forward_kwargs)

def ParseOutputLayer(y_4d, subset=None):
    if len(y_4d.shape) == 2:
        if subset is None:
            return (y_4d, [])
        return y_4d

    num_data = y_4d.shape[0] * (y_4d.shape[2]-2) * (y_4d.shape[3]-2)
    ndim = y_4d.shape[1]
    y_4d_nhwc = y_4d.swapaxes(1, 3).swapaxes(1, 2)
    y_2d = y_4d_nhwc[:, 1:y_4d.shape[2]-1, 1:y_4d.shape[3]-1, :].reshape((num_data, ndim))

    if subset is None:
        subset = np.random.permutation(num_data)[:sample_per_image]
        return (y_2d[subset], subset)
    return y_2d[subset]

def TrainingImageNames(dataset_filename, do_shuffle=True):
    image_names = [line.strip().split(' ')[0] for line in open(dataset_filename, 'r').readlines()]
    if do_shuffle:
        shuffle(image_names)
    return image_names

def MakeTrainingDataSub(image_name, size, mean_bgr, layer, true_net, compressed_net):
    sample = PreprocessImage(image_name, size, mean_bgr)

    Forward(true_net, sample, layer)
    Forward(compressed_net, sample, layer + '_kw')

    y_true, subset = ParseOutputLayer(true_net.blobs[layer].data)
    y_dirty = ParseOutputLayer(compressed_net.blobs[layer + '_kw'].data, subset)

    return (y_true, y_dirty)

def MakeTrainingDataset(cfgs, layer, true_net, compressed_net):
    image_names = TrainingImageNames(cfgs['data'])
    if len(image_names) > num_train_images:
        image_names = image_names[:num_train_images]
    size = cfgs['input_size']
    mean_bgr = cfgs['mean_img']

    y_true, y_dirty = MakeTrainingDataSub(image_names[0], size, mean_bgr, layer, true_net, compressed_net)
    Y_true = np.zeros((len(image_names), y_true.shape[0], y_true.shape[1]), dtype=np.float32)
    Y_dirty = np.zeros((len(image_names), y_true.shape[0], y_true.shape[1]), dtype=np.float32)
    Y_true[0] = y_true
    Y_dirty[0] = y_dirty

    start_time = datetime.datetime.now()
    for i in range(1, len(image_names)):
        Y_true[i], Y_dirty[i] = MakeTrainingDataSub(image_names[i], size, mean_bgr, layer, true_net, compressed_net)

        if (i+1) % 10 == 0:
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print '[%6d image processed] Time=%.2fs' % (i+1, elapsed_time)
            start_time = end_time

    Y_true = Y_true.reshape((Y_true.shape[0] * Y_true.shape[1], Y_true.shape[2])).swapaxes(0, 1).astype(np.float64, copy=True)
    Y_dirty = Y_dirty.reshape((Y_dirty.shape[0] * Y_dirty.shape[1], Y_dirty.shape[2])).swapaxes(0, 1).astype(np.float64, copy=True)

    return (Y_true, Y_dirty)

def ConvCompressionError(Y_true, Y_dirty, M, b):
    Y = Y_true.copy()
    Y[Y < 0] = 0

    Z = np.dot(M, Y_dirty)
    for i in range(Z.shape[0]):
        Z[i] += b[i]
    Z[Z < 0] = 0

    err = ((Y - Z) ** 2).sum()
    yTz = np.multiply(Y, Z).sum(axis=0)
    y_norm = np.sqrt((Y ** 2).sum(axis=0))
    z_norm = np.sqrt((Z ** 2).sum(axis=0))
    cossim = np.divide(yTz, 1e-10 + np.multiply(y_norm, z_norm)).mean()
    return (err, cossim)

def CompressConvLayerAuxiliary(Y, Y_dirty, M, b, penalty):
    Y_approx = np.dot(M, Y_dirty)
    for i in range(Y.shape[0]):
        Y_approx[i] += b[i]
    Y_approx = Y_approx.reshape((np.prod(Y.shape),))

    Z_minus = Y_approx.copy()
    Z_minus[Z_minus > 0] = 0

    Y_nonlinear = Y.reshape((np.prod(Y.shape),)).copy()
    Y_nonlinear[Y_nonlinear < 0] = 0

    Z_plus = (penalty * Y_approx + Y_nonlinear) / (penalty + 1)
    Z_plus[Z_plus < 0] = 0

    err_minus = (Y_nonlinear ** 2) \
                   + penalty * ((Z_minus - Y_approx) ** 2)
    err_plus = ((Y_nonlinear - Z_plus) ** 2) \
                   + penalty * ((Z_plus - Y_approx) ** 2)
    minus_is_better = err_minus < err_plus
    Z_plus[minus_is_better] = Z_minus[minus_is_better]
    return Z_plus.reshape((Y.shape[0], Y.shape[1]))

def CompressConvLayerSub(Y_true, Y_dirty, rank, penalty=None, M=None, b=None):
    start_time = datetime.datetime.now()

    if penalty is not None:
        Z = CompressConvLayerAuxiliary(Y_true, Y_dirty, M, b, penalty)
    else:
        Z = Y_true.copy()

    mu_Z = Z.mean(axis=1)
    for i in range(Z.shape[0]):
        Z[i] -= mu_Z[i]

    Y = Y_dirty.copy()
    mu_Y = Y.mean(axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= mu_Y[i]

    ZY = np.tensordot(Z, Y, (1, 1))
    M_new = np.dot(ZY, np.linalg.inv(np.tensordot(Y, Y, (1, 1))))

    D, U = np.linalg.eigh(np.tensordot(ZY, M_new, (1, 1)))
    S = np.sqrt(D[-rank:])
    U = U[:, -rank:]
    SinvUT = np.tensordot(np.diag(1.0 / S), U, (1, 1))
    VT = np.dot(SinvUT, M_new)
    Ssqrt = np.diag(np.sqrt(S))

    P = np.dot(U, Ssqrt)
    QT = np.dot(Ssqrt, VT)
    M_new = np.dot(P, QT)
    b_new = mu_Y - np.dot(M_new, mu_Y)

    err, cossim = ConvCompressionError(Y_true, Y_dirty, M_new, b_new)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Sum of squared errors = %.4f, Average cosine similarity = %.4f, Elapsed time = %.2f' \
          % (err, cossim, elapsed_time)
    return (M_new, b_new, P, QT)

def CompressConvLayer(Y_true, Y_dirty, W_true, b_true, rank):
    M, b, P, QT = CompressConvLayerSub(Y_true, Y_dirty, rank)
    for i in range(5):
        M, b, P, QT = CompressConvLayerSub(Y_true, Y_dirty, rank, 0.01, M, b)
    for i in range(5):
        M, b, P, QT = CompressConvLayerSub(Y_true, Y_dirty, rank, 0.1, M, b)
    for i in range(5):
        M, b, P, QT = CompressConvLayerSub(Y_true, Y_dirty, rank, 10.0, M, b)
    for i in range(5):
        M, b, P, QT = CompressConvLayerSub(Y_true, Y_dirty, rank, 100.0, M, b)

    W_2d = W_true.reshape((W_true.shape[0], np.prod(W_true.shape[1:])))
    b_1d = b_true.reshape((b_true.shape[0],))

    W1 = np.dot(QT, W_2d).astype(np.float32, copy=True)
    b1 = np.dot(QT, b_1d).astype(np.float32, copy=True)
    W2 = P.astype(np.float32, copy=True)
    b2 = b.astype(np.float32, copy=True)
    if len(W_true.shape) == 4:
        W1 = W1.reshape((QT.shape[0], W_true.shape[1], W_true.shape[2], W_true.shape[3]))
        W2 = W2.reshape((P.shape[0], P.shape[1], 1, 1))

    return (W1, b1, W2, b2)

def CompressFCLayerSub(W, rank):
    start_time = datetime.datetime.now()

    if W.shape[0] == W.shape[1]:
        U, S, V = np.linalg.svd(W)
        S_sqrt = np.diag(np.sqrt(S[:rank]))
        P = np.dot(U[:, :rank], S_sqrt)
        QT = np.tensordot(S_sqrt, V[:, :rank], (1, 1))
        energy = (S[:rank] ** 2).sum() / (S ** 2).sum()

    else:
        if W.shape[0] > W.shape[1]:
            W = W.transpose().copy()
        WWT = np.tensordot(W, W, (1, 1))
        D, U = np.linalg.eigh(WWT)
        U = U[:, -rank:]
        S_inv = np.diag(1.0 / np.sqrt(D[-rank:]))
        VT = np.dot(np.tensordot(S_inv, U, (1, 1)), W)
        S_sqrt = np.diag(np.sqrt(np.sqrt(D[-rank:])))
        P = np.dot(U, S_sqrt)
        QT = np.dot(S_sqrt, VT)
        energy = D[-rank:].sum() / D.sum()

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Truncated SVD Accumulative Energy = %.4f' % energy
    return (P, QT)

def CompressFCLayer(W_true, b_true, rank):
    P, QT = CompressFCLayerSub(W_true, rank)

    W1 = QT.astype(np.float32, copy=True)
    b1 = np.zeros((QT.shape[0],), dtype=np.float32)
    W2 = P.astype(np.float32, copy=True)
    b2 = b_true.copy()

    return (W1, b1, W2, b2)

def CompressConvKernelSub(W, rank, H=None, G=None):
    start_time = datetime.datetime.now()

    if H is None:
        H = np.random.normal(size=(W.shape[1]*W.shape[2], rank))

    W_G = W.swapaxes(2, 3).swapaxes(1, 2).reshape((W.shape[0]*W.shape[3], W.shape[1]*W.shape[2]))
    W_H = W.swapaxes(0, 1).swapaxes(1, 2).reshape((W.shape[1]*W.shape[2], W.shape[0]*W.shape[3]))
    G = np.dot(W_G, np.dot(H, np.linalg.inv(np.tensordot(H, H, (0, 0)))))
    H = np.dot(W_H, np.dot(G, np.linalg.inv(np.tensordot(G, G, (0, 0)))))

    err = ((W_G - np.tensordot(G, H, (1, 1))) ** 2).sum()
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Kernel approximation error = %.4f, Elapsed time = %.2f' % (err, elapsed_time)
    return (H, G)

def CompressConvKernel(W_true, b_true, rank):
    H, G = CompressConvKernelSub(W_true, rank)
    for i in range(1, 30):
        H, G = CompressConvKernelSub(W_true, rank, H, G)

    W1 = H.reshape((1, W_true.shape[1], W_true.shape[2], rank)).swapaxes(0, 3).astype(np.float32, copy=True)
    b1 = np.zeros((rank,), dtype=np.float32)
    W2 = G.reshape((W_true.shape[0], W_true.shape[3], 1, rank)).swapaxes(1, 3).astype(np.float32, copy=True)
    b2 = b_true.copy()

    return (W1, b1, W2, b2)

def UpdateLayerParams(net, layer, W, b):
    net.params[layer][0].data[:] = W
    net.params[layer][1].data[:] = b

def RunSub(cfgs, layer, rank, true_net, compressed_net):
    W_true = true_net.params[layer][0].data
    b_true = true_net.params[layer][1].data

    if 'conv' in layer:
        W1, b1, W2, b2 = CompressConvKernel(W_true, b_true, np.prod(b_true.shape))
        UpdateLayerParams(compressed_net, layer + '_kh', W1, b1)
        UpdateLayerParams(compressed_net, layer + '_kw', W2, b2)

        Y_true, Y_dirty = MakeTrainingDataset(cfgs, layer, true_net, compressed_net)
        W1, b1, W2, b2 = CompressConvLayer(Y_true, Y_dirty, W2, b2, rank)
        UpdateLayerParams(compressed_net, layer + '_kw_a', W1, b1)
        UpdateLayerParams(compressed_net, layer + '_kw_b', W2, b2)

    if 'fc' in layer:
        W1, b1, W2, b2 = CompressFCLayer(W_true, b_true, rank)
        UpdateLayerParams(compressed_net, layer + '_a', W1, b1)
        UpdateLayerParams(compressed_net, layer + '_b', W2, b2)

def Run(cfgs):
    true_net = caffe.Net('prototxt/true.prototxt', 'caffemodel/true.caffemodel', caffe.TEST)
    compressed_net = caffe.Net('prototxt/compressed_3d.prototxt', 'caffemodel/true.caffemodel', caffe.TEST)

    for i in range(len(cfgs['layers'])):
        layer, rank = cfgs['layers'][i]
        print '########### Compressing %s to rank %d ###########' % (layer, rank)
        RunSub(cfgs, layer, rank, true_net, compressed_net)

    compressed_net.save('caffemodel/compressed_3d.caffemodel')

import config
import caffe

if __name__ == "__main__":
    cfgs = config.model_['vggnet']

    caffe.set_mode_gpu()
    caffe.set_device(0)

    Run(cfgs)
