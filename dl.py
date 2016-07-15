import numpy as np


def load_data(filename):
  from struct import pack, unpack
  f = open(filename, 'rb')
  ndim = unpack("i", f.read(4))[0]
  shape = np.frombuffer(f.read(ndim * 4), dtype=np.int32, count=-1)
  data = np.fromfile(f, dtype=np.float32).reshape(shape)
  f.close()
  return data


def save_data(filename, data):
  from struct import pack, unpack
  f = open(filename, 'wb')
  ndim = len(data.shape)
  f.write(pack('i', ndim))
  f.write(pack('i' * ndim, *data.shape))
  data.tofile(f)
  f.close()


def load_inception(proto_src, proto_dest, model_src):
  import caffe
  caffe.set_mode_cpu()
  #caffe.set_device(1)
  net_src = caffe.Net(proto_src, model_src, caffe.TEST)
  net_dest = caffe.Net(proto_dest, caffe.TEST)
  for key in net_src.params.keys():
    for i in range(len(net_src.params[key])):
      net_dest.params[key][i].data[:] = net_src.params[key][i].data[:]
  return net_dest


def combine_conv_bn_scale_pva33(net):
  for name in net._layer_names:
    if name.endswith('/bn'):
      key_conv = name[:-3]
      key_bn = name
      key_scale = name[:-3] + '/scale'
    else:
      continue
    combine_conv_bn_scale(net, key_conv, key_bn, key_scale)


def combine_conv_bn_scale_pva900(net):
  for name in net._layer_names:
    if name.endswith('/bn'):
      if name.startswith('fc'):
        key_conv = name[:-3]
        key_bn = name
        key_scale = name[:-3] + '/scale'
      else:
        key_conv = name[:-3] + '/conv'
        key_bn = name
        key_scale = name + '_scale'
    elif name.endswith('/proj_bn'):
      key_conv = name[:-3]
      key_bn = name
      key_scale = name + '_scale'
    else:
      continue
    combine_conv_bn_scale(net, key_conv, key_bn, key_scale)


def combine_conv_bn_scale(net, key_conv, key_bn, key_scale):
  def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)

  if not net.params.has_key(key_bn):
    print 'No batch norm layer {:s} to be combined'.format(key_bn)
    return
  if not net.params.has_key(key_conv):
    print '[ERROR] Cannot find conv layer {:s}'.format(key_conv)
    return

  bn_mean = copy_double(net.params[key_bn][0].data)
  bn_variance = copy_double(net.params[key_bn][1].data)
  num_bn_samples = copy_double(net.params[key_bn][2].data)
  net.params[key_bn][0].data[:] = 0
  net.params[key_bn][1].data[:] = 1
  net.params[key_bn][2].data[:] = 1
  if num_bn_samples[0] == 0:
    num_bn_samples[0] = 1

  if net.params.has_key(key_scale):
    print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)
    scale_weight = copy_double(net.params[key_scale][0].data)
    scale_bias = copy_double(net.params[key_scale][1].data)
    net.params[key_scale][0].data[:] = 1
    net.params[key_scale][1].data[:] = 0
  else:
    print 'Combine {:s} + {:s}'.format(key_conv, key_bn)
    scale_weight = 1
    scale_bias = 0

  weight = copy_double(net.params[key_conv][0].data)
  bias = copy_double(net.params[key_conv][1].data)
  alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
  net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
  for i in range(len(alpha)):
    net.params[key_conv][0].data[i] = weight[i] * alpha[i]


def truncated_svd(W_true, rank):
  import datetime
  start_time = datetime.datetime.now()

  if W_true.shape[0] > W_true.shape[1]:
    W = W_true.transpose().copy()
  else:
    W = W_true.copy()

  WWT = np.tensordot(W, W, (1, 1))
  D, U = np.linalg.eigh(WWT)
  U = U[:, -rank:]
  S_inv = np.diag(1.0 / np.sqrt(D[-rank:]))
  VT = np.dot(np.tensordot(S_inv, U, (1, 1)), W)
  S_sqrt = np.diag(np.sqrt(np.sqrt(D[-rank:])))

  if W_true.shape[0] > W_true.shape[1]:
    P = np.dot(VT.transpose(), S_sqrt)
    Q = np.dot(S_sqrt, U.transpose())
  else:
    P = np.dot(U, S_sqrt)
    Q = np.dot(S_sqrt, VT)
  energy = D[-rank:].sum() / D.sum()

  elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
  print 'Truncated SVD Accumulative Energy = %.4f' % energy
  return (P, Q)


def compress_fc(net, key_fc):
  def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)

  key_L = key_fc + '_L'
  key_U = key_fc + '_U'
  if not net.params.has_key(key_fc):
    print '[ERROR] Cannot find fc layer {:s}'.format(key_fc)
    return
  if not net.params.has_key(key_L):
    print '[ERROR] Cannot find fc compression layer {:s}'.format(key_L)
    return
  if not net.params.has_key(key_U):
    print '[ERROR] Cannot find fc reconstruction layer {:s}'.format(key_U)
    return

  W_true = copy_double(net.params[key_fc][0].data)
  b_true = copy_double(net.params[key_fc][1].data)
  true_rank = net.params[key_fc][0].data.shape[0]
  rank = net.params[key_L][0].data.shape[0]

  print 'Compress {:s} ({:d} dim) -> {:s} ({:d} dim) + {:s}'.format(key_fc, true_rank, key_L, rank, key_U)
  P, Q = truncated_svd(W_true, rank)
  W1 = Q.astype(np.float32, copy=True)
  b1 = np.zeros((Q.shape[0],), dtype=np.float32)
  W2 = P.astype(np.float32, copy=True)
  b2 = b_true.copy()

  net.params[key_L][0].data[:] = W1
  net.params[key_L][1].data[:] = b1
  net.params[key_U][0].data[:] = W2
  net.params[key_U][1].data[:] = b2


def save_bin(net, path):
  for name in net.params.keys():
    for param_id in range(len(net.params[name])):
      key = '{:s}/{:s}_param{:d}.bin'.format(path, name.replace('/', '_'), param_id)
      save_data(key, net.params[name][param_id].data)


def parse_args():
  import sys, argparse
  parser = argparse.ArgumentParser(description='Non-zero shift bug fix for PVA-9.0.0 model')
  parser.add_argument('--proto_src', dest='proto_src',
                      help='original PVA-9.0.0 prototxt',
                      default=None, type=str)
  parser.add_argument('--model_src', dest='model_src',
                      help='original PVA-9.0.0 caffemodel',
                      default=None, type=str)
  parser.add_argument('--proto_dest', dest='proto_dest',
                      help='new PVA-9.0.0 prototxt',
                      default=None, type=str)
  parser.add_argument('--model_dest', dest='model_dest',
                      help='new PVA-9.0.0 caffemodel',
                      default=None, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = parse_args()
  print('Called with args:')
  print(args)

  net = load_inception(args.proto_src, args.proto_dest, args.model_src)
  combine_conv_bn_scale_pva900(net)
  compress_fc(net, 'fc6')
  compress_fc(net, 'fc7')
  net.save(args.model_dest)
