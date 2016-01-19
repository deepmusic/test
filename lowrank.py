import numpy as np
import os
import datetime
import test

weight_postfix = '_weight.npy'
bias_postfix = '_bias.npy'
data_approx_postfix = '_approx'

num_train_images = 3000
sample_per_image = 16
max_num_data = 50000

def ParseInputLayer(x_4d_chw, k_h, k_w, s_h=1, s_w=1):
    d1_h = -int(np.floor((k_h - 1) / 2.0))
    d2_h = int(np.ceil((k_h - 1) / 2.0)) + 1
    d1_w = -int(np.floor((k_w - 1) / 2.0))
    d2_w = int(np.ceil((k_w - 1) / 2.0)) + 1
    range_row = range(1, x_4d_chw.shape[2]-1, s_h)
    range_col = range(1, x_4d_chw.shape[3]-1, s_w)

    num_data = len(range_row) * len(range_col)
    ndim = x_4d_chw.shape[1] * k_h * k_w
    x_2d = np.zeros((num_data, ndim), dtype=np.float32)

    data_count = 0
    for row in range_row:
        for col in range_col:
            x_1d = x_4d_chw[0, :, row+d1_h:row+d2_h, col+d1_w:col+d2_w]
            x_2d[data_count] = x_1d.reshape((ndim,))
            data_count += 1
    return x_2d

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

def MakeOutputData(image_name, mean_img, layer, net_true, net_approx):
    sample = test.PreprocessImage(image_name, mean_img)
    test.Forward(net_true, sample, layer)
    test.Forward(net_approx, sample, layer + 'b')
    y_true, subset = ParseOutputLayer(net_true.blobs[layer].data)
    y_approx = ParseOutputLayer(net_approx.blobs[layer + 'b'].data, subset)
    return (y_true, y_approx)

def MakeOutputDataset(config, layer, net_true, net_approx):
    image_names = config.TrainImageNames()
    if len(image_names) > num_train_images:
        image_names = image_names[:num_train_images]
    mean_img = config.mean_img

    y, y_approx = MakeOutputData(image_names[0], mean_img, layer, net_true, net_approx)
    Y_true = np.zeros((len(image_names), y.shape[0], y.shape[1]), dtype=np.float32)
    Y_approx = np.zeros((len(image_names), y.shape[0], y.shape[1]), dtype=np.float32)
    Y_true[0] = y
    Y_approx[0] = y_approx

    start_time = datetime.datetime.now()
    for i in range(1, len(image_names)):
        Y_true[i], Y_approx[i] = \
                MakeOutputData(image_names[i], mean_img, layer, net_true, net_approx)

        if (i+1) % 100 == 0:
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print '[%6d image processed] Time=%.2fs' % (i+1, elapsed_time)
            start_time = end_time
    return (Y_true, Y_approx)

def LoadOutputDataset(path, layer, postfix=''):
    Y = np.load(os.path.join(path, layer + postfix + '.npy'))
    Y = np.asarray(Y.reshape((Y.shape[0] * Y.shape[1], Y.shape[2])).swapaxes(0, 1), dtype=np.float64)
    return Y

def SVD(W, rank):
    start_time = datetime.datetime.now()

    if W.shape[0] == W.shape[1]:
        U, S, V = np.linalg.svd(W)
        S_sqrt = np.diag(np.sqrt(S[:rank]))
        P = np.dot(U[:, :rank], S_sqrt)
        QT = np.tensordot(S_sqrt, V[:, :rank], (1, 1))
        energy = (S[:rank] ** 2).sum() / (S ** 2).sum()

    else:
        if W.shape[0] > W.shape[1]:
            W = W.transpose()
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
    print 'PCA Accumulative Energy = %.4f' % energy
    return (P, QT)

def LinearLowRank(path, layer, rank):
    start_time = datetime.datetime.now()
    Y = LoadOutputDataset(path, layer)
    if Y.shape[1] > max_num_data:
        subset = np.random.permutation(Y.shape[1])[:max_num_data]
        Y = Y[:, subset]

    mu = Y.mean(axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= mu[i]

    YYT = np.tensordot(Y, Y, (1, 1))
    D, U = np.linalg.eigh(YYT)
    print 'PCA Accumulative Energy = %.4f' % (D[-rank:].sum() / D.sum())

    M = np.dot(U[:, -rank:], U[:, -rank:].transpose())
    b = mu - np.dot(M, mu)

    err, cossim = NonlinearError(path, layer, M, b)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Sum of squared errors = %.4f, Average cosine similarity = %.4f, Elapsed time = %.2f' \
          % (err, cossim, elapsed_time)
    return (M, b)

def NonlinearLowRankAuxiliary(Y, Y_approx, M, b, penalty):
    Y_approx = np.dot(M, Y_approx)
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

def NonlinearLowRankSub(path, layer, rank, penalty=None, M=None, b=None):
    start_time = datetime.datetime.now()

    Y_true = LoadOutputDataset(path, layer)
    Y = LoadOutputDataset(path, layer, data_approx_postfix)
    if Y.shape[1] > max_num_data:
        subset = np.random.permutation(Y.shape[1])[:max_num_data]
        Y_true = Y_true[:, subset]
        Y = Y[:, subset]

    if penalty is not None:
        Z = NonlinearLowRankAuxiliary(Y_true, Y, M, b, penalty)
    else:
        Z = Y_true

    mu_Z = Z.mean(axis=1)
    for i in range(Z.shape[0]):
        Z[i] -= mu_Z[i]
    mu_Y = Y.mean(axis=1)
    for i in range(Y.shape[0]):
        Y[i] -= mu_Y[i]

    #YYT = np.tensordot(Y, Y, (1, 1))
    #D, U = np.linalg.eigh(YYT)
    #rank_YYT = (D > 0).sum()
    #D = D[-rank_YYT:]
    #U = U[:, -rank_YYT:]
    #V_YYT = np.dot(U, np.diag(1.0 / np.sqrt(D)))
    #YYT_inv = np.tensordot(V_YYT, V_YYT, (1, 1))

    ZY = np.tensordot(Z, Y, (1, 1))
    M_new = np.dot(ZY, np.linalg.inv(np.tensordot(Y, Y, (1, 1))))
    #M_new = np.dot(ZY, YYT_inv)

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

    err, cossim = NonlinearError(path, layer, M_new, b_new)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print 'Sum of squared errors = %.4f, Average cosine similarity = %.4f, Elapsed time = %.2f' \
          % (err, cossim, elapsed_time)
    return (M_new, b_new, P, QT)

def NonlinearError(path, layer, M, b):
    Y = LoadOutputDataset(path, layer)
    Y_approx = LoadOutputDataset(path, layer, data_approx_postfix)
    if Y.shape[1] > max_num_data:
        subset = np.random.permutation(Y.shape[1])[:max_num_data]
        Y = Y[:, subset]
        Y_approx = Y_approx[:, subset]

    Z = np.dot(M, Y_approx)
    for i in range(Z.shape[0]):
        Z[i] += b[i]

    Y[Y < 0] = 0
    Z[Z < 0] = 0

    err = ((Y - Z) ** 2).sum()
    yTz = np.multiply(Y, Z).sum(axis=0)
    y_norm = np.sqrt((Y ** 2).sum(axis=0))
    z_norm = np.sqrt((Z ** 2).sum(axis=0))
    cossim = np.divide(yTz, 1e-10 + np.multiply(y_norm, z_norm)).mean()
    return (err, cossim)

def NonlinearLowRank(path, layer, rank):
    M, b, P, QT = NonlinearLowRankSub(path, layer, rank)
    for i in range(5):
        M, b, P, QT = NonlinearLowRankSub(path, layer, rank, 0.01, M, b)
    for i in range(5):
        M, b, P, QT = NonlinearLowRankSub(path, layer, rank, 1.0, M, b)
    for i in range(5):
        M, b, P, QT = NonlinearLowRankSub(path, layer, rank, 10.0, M, b)
    for i in range(5):
        M, b, P, QT = NonlinearLowRankSub(path, layer, rank, 100.0, M, b)
    return (M, b, P, QT)

def SaveLowRankParams(path, layer, b, P, QT):
    W_4d = np.load(os.path.join(path, layer + 'b' + weight_postfix))
    b_4d = np.load(os.path.join(path, layer + 'b' + bias_postfix))
    W_2d = W_4d.reshape((W_4d.shape[0], np.prod(W_4d.shape[1:])))
    b_1d = b_4d.reshape((b_4d.shape[0],))

    W1_low = np.asarray(np.dot(QT, W_2d), dtype=np.float32)
    b1_low = np.asarray(np.dot(QT, b_1d), dtype=np.float32)
    W2_low = np.asarray(P, dtype=np.float32)
    b2_low = np.asarray(b, dtype=np.float32)
    if len(W_4d.shape) == 4:
        W1_low = W1_low.reshape((QT.shape[0], W_4d.shape[1], W_4d.shape[2], W_4d.shape[3]))
        W2_low = W2_low.reshape((P.shape[0], P.shape[1], 1, 1))

    np.save(os.path.join(path, layer + 'b1' + weight_postfix), W1_low)
    np.save(os.path.join(path, layer + 'b1' + bias_postfix), b1_low)
    np.save(os.path.join(path, layer + 'b2' + weight_postfix), W2_low)
    np.save(os.path.join(path, layer + 'b2' + bias_postfix), b2_low)

def SaveFCParams(path, layer, P, QT):
    W_2d = np.load(os.path.join(path, layer + weight_postfix))
    b_1d = np.load(os.path.join(path, layer + bias_postfix))

    W1_low = np.asarray(QT, dtype=np.float32)
    b1_low = np.zeros((QT.shape[0],), dtype=np.float32)
    W2_low = np.asarray(P, dtype=np.float32)
    b2_low = b_1d

    np.save(os.path.join(path, layer + '_1' + weight_postfix), W1_low)
    np.save(os.path.join(path, layer + '_1' + bias_postfix), b1_low)
    np.save(os.path.join(path, layer + '_2' + weight_postfix), W2_low)
    np.save(os.path.join(path, layer + '_2' + bias_postfix), b2_low)

def PrepareDataset(config, layer, net_true, net_approx):
    Y_true, Y_approx = MakeOutputDataset(config, layer, net_true, net_approx)
    np.save(os.path.join(config.dump_path, layer), Y_true)
    np.save(os.path.join(config.dump_path, layer + data_approx_postfix), Y_approx)

def Run(config, layer, rank, net_true, net_approx):
    if 'conv' in layer:
        LowRankKernel(config.dump_path, layer, rank)
        UpdateNetLayerParams(config.dump_path, layer, net_approx, ['a', 'b'])
        PrepareDataset(config, layer, net_true, net_approx)
        M, b, P, QT = NonlinearLowRank(config.dump_path, layer, rank)
        SaveLowRankParams(config.dump_path, layer, b, P, QT)

    if 'fc' in layer:
        W_2d = np.load(os.path.join(config.dump_path, layer + weight_postfix))
        P, QT = SVD(W_2d, rank)
        SaveFCParams(config.dump_path, layer, P, QT)

def UpdateNetLayerParams(dump_path, layer, net, layer_postfixs=None):
    if layer_postfixs is None:
        if 'conv' in layer:
            layer_postfixs = ['a', 'b1', 'b2']
        else:
            layer_postfixs = ['_1', '_2']

    for layer_postfix in layer_postfixs:
        weight = np.load(os.path.join(dump_path, layer + layer_postfix + weight_postfix))
        bias = np.load(os.path.join(dump_path, layer + layer_postfix + bias_postfix))
        net.params[layer + layer_postfix][0].data[:] = weight.copy()
        net.params[layer + layer_postfix][1].data[:] = bias.copy()

def DumpParams(dump_path, net):
    layers = net.params.keys()
    for layer in layers:
        np.save(os.path.join(dump_path, layer + weight_postfix), net.params[layer][0].data)
        np.save(os.path.join(dump_path, layer + bias_postfix), net.params[layer][1].data)

def LowRankKernelSub(W, rank, H=None, G=None):
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

def LowRankKernel(dump_path, layer, rank):
    W_4d = np.load(os.path.join(dump_path, layer + weight_postfix))
    b_1d = np.load(os.path.join(dump_path, layer + bias_postfix))

    H, G = LowRankKernelSub(W_4d, rank)
    for i in range(1, 30):
        H, G = LowRankKernelSub(W_4d, rank, H, G)

    W1_low = H.reshape((1, W_4d.shape[1], W_4d.shape[2], rank)).swapaxes(0, 3).copy()
    W1_low = np.asarray(W1_low, dtype=np.float32)
    b1_low = np.zeros((rank,), dtype=np.float32)
    W2_low = G.reshape((W_4d.shape[0], W_4d.shape[3], 1, rank)).swapaxes(1, 3).copy()
    W2_low = np.asarray(W2_low, dtype=np.float32)
    b2_low = np.asarray(b_1d, dtype=np.float32)

    np.save(os.path.join(dump_path, layer + 'a' + weight_postfix), W1_low)
    np.save(os.path.join(dump_path, layer + 'a' + bias_postfix), b1_low)
    np.save(os.path.join(dump_path, layer + 'b' + weight_postfix), W2_low)
    np.save(os.path.join(dump_path, layer + 'b' + bias_postfix), b2_low)
