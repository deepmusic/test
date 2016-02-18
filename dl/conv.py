import mxnet as mx
import numpy as np

data_in = mx.symbol.Variable(name='data')
conv = mx.symbol.Convolution(name='conv', data=data_in, num_filter=100,
                             kernel=(3, 3), stride=(1, 1), pad=(1, 1))

X_size = (10, 100, 5, 5)
W_size = (100, 100, 3, 3)
b_size = (100,)
try:
  X = np.loadtxt('X.txt', dtype=np.float32).reshape(X_size)
except Exception:
  X = np.asarray(np.random.normal(0, 1, np.prod(X_size)), dtype=np.float32)
  np.savetxt('X.txt', X, fmt='%.6f', delimiter=' ')
  X = X.reshape(X_size)
try:
  W = np.loadtxt('W.txt', dtype=np.float32).reshape(W_size)
except Exception:
  W = np.asarray(np.random.normal(0, 0.02, np.prod(W_size)), dtype=np.float32)
  np.savetxt('W.txt', W, fmt='%.6f', delimiter=' ')
  W = W.reshape(W_size)
try:
  b = np.loadtxt('b.txt', dtype=np.float32).reshape(b_size)
except Exception:
  b = np.asarray(np.random.normal(0, 0.01, np.prod(b_size)), dtype=np.float32)
  np.savetxt('b.txt', b, fmt='%.6f', delimiter=' ')
  b = b.reshape(b_size)

arg_params = {}
arg_params['conv_weight'] = mx.nd.zeros(W.shape)
arg_params['conv_weight'][:] = mx.nd.array(W)
arg_params['conv_bias'] = mx.nd.zeros(b.shape)
arg_params['conv_bias'][:] = mx.nd.array(b)

model = mx.model.FeedForward(ctx=mx.gpu(1), symbol=conv, arg_params=arg_params,
                             aux_params={}, num_epoch=1,
                             learning_rate=0.01, momentum=0.9, wd=0.0001)

for i in range(100):
  Y = model.predict(mx.io.NDArrayIter([X], batch_size=10))
  X = model.predict(mx.io.NDArrayIter([Y], batch_size=10))
for n in range(Y.shape[0]):
  print 'Y[{:d}] ({:d} x {:d} x {:d})'.format(n, Y.shape[1], Y.shape[2], Y.shape[3])
  for c in range(Y.shape[1]):
    for h in range(Y.shape[2]):
      outstr = ''
      for w in range(Y.shape[3]):
        outstr += '{:03.5f} '.format(Y[n,c,h,w])
      print outstr
    print '\n'
  print '\n\n===============================\n'
