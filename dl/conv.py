import mxnet as mx
import numpy as np

data_in = mx.symbol.Variable(name='data')
conv = mx.symbol.Convolution(name='conv', data=data_in, num_filter=5,
                             kernel=(3, 3), stride=(2, 2), pad=(1, 1))

X = np.loadtxt('X.txt').reshape(2, 10, 5, 5)
W = np.loadtxt('W.txt').reshape(5, 10, 3, 3)
b = np.array([0.1, -0.1, 0.2, -0.2, 0], dtype=np.float32)

arg_params = {}
arg_params['conv_weight'] = mx.nd.zeros(W.shape)
arg_params['conv_weight'][:] = mx.nd.array(W)
arg_params['conv_bias'] = mx.nd.zeros(b.shape)
arg_params['conv_bias'][:] = mx.nd.array(b)

model = mx.model.FeedForward(ctx=mx.cpu(), symbol=conv, arg_params=arg_params,
                             aux_params={}, num_epoch=1,
                             learning_rate=0.01, momentum=0.9, wd=0.0001)

Y = model.predict(mx.io.NDArrayIter([X], batch_size=2))
print Y
