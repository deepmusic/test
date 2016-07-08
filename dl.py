from struct import pack, unpack
import numpy as np
import scipy.ndimage
import scipy.misc

def load_inception():
  #proto = '../new-faster-rcnn/pva_inception2_3_coco.pt'
  #model = '../new-faster-rcnn/output/faster_rcnn_once_25anc_plus/pvtdb_pengo_80_pva/pva_inception2_3_once_iter_330000.caffemodel'
  #proto = '../new-faster-rcnn/pva_inception2_coco.pt'
  #model = '../new-faster-rcnn/output/faster_rcnn_once_25anc_plus/pvtdb_pengo_80_pva/pva_inception2_2_once_iter_4720000.caffemodel'
  #proto = '../new-faster-rcnn/pva7.1.1_coco.pt'
  #model = '../new-faster-rcnn/pva7.1.1_coco_once_iter_880000.caffemodel'
  #proto = '../new-faster-rcnn/pva_inception2_4.pt'
  #model = '../new-faster-rcnn/output/faster_rcnn_once_25anc_plus/pvtdb_pengo3_24_pva/pva_inception2_4_once_iter_1750000.caffemodel'
  #proto = '../new-faster-rcnn/models/pva_inception64_4/pva_inception64_4_test.pt'
  #model = '../new-faster-rcnn/models/pva_inception64_4/pva_inception64_4_train_iter_1074505.caffemodel'
  #proto = '../new-faster-rcnn/models/pva_inception64_4/faster_rcnn_once/faster_rcnn_train_convert.pt'
  #model = '../new-faster-rcnn/output/faster_rcnn_once_25anc_plus/pvtdb_pengo2_84_pva/Round3/pva_inception64_4_once_iter_900000.caffemodel'
  proto_src = 'faster_rcnn_train_test.pt'
  proto_comb = 'faster_rcnn_train_test_comb.pt'
  model = 'pva9.0.0_fixed.caffemodel'
  import caffe
  caffe.set_mode_cpu()
  #caffe.set_device(1)
  net_src = caffe.Net(proto_src, model, caffe.TEST)
  net_dest = caffe.Net(proto_comb, caffe.TEST)
  for key in net_src.params.keys():
    for i in range(len(net_src.params[key])):
      net_dest.params[key][i].data[:] = net_src.params[key][i].data[:]
  return net_dest

def test_bn(net):
  a = net.blobs['inc3a/conv5_2'].data.copy()
  b = net.blobs['inc3a/conv5_2/bn'].data.copy()
  n = net.params['inc3a/conv5_2/bn'][2].data[0]
  m = net.params['inc3a/conv5_2/bn'][0].data.copy()
  v = net.params['inc3a/conv5_2/bn'][1].data.copy()
  m /= n
  v = 1 / np.sqrt(v / n)
  for i in range(m.shape[0]):
    a[:,i,:,:] = (a[:,i,:,:] - m[i]) * v[i]

def test_scale(net):
  a = net.blobs['inc3a/conv5_2/bn'].data.copy()
  b = net.blobs['inc3a/conv5_2/scale'].data.copy()
  w = net.params['inc3a/conv5_2/scale'][0].data.copy()
  c = net.params['inc3a/conv5_2/scale'][1].data.copy()
  for i in range(w.shape[0]):
    a[:,i,:,:] = w[i] * a[:,i,:,:] + c[i]

def test_bn_scale(net):
  a = net.blobs['inc3a/conv5_2'].data.copy()
  b = net.blobs['inc3a/conv5_2/scale'].data.copy()
  n = net.params['inc3a/conv5_2/bn'][2].data[0]
  m = net.params['inc3a/conv5_2/bn'][0].data.copy()
  v = net.params['inc3a/conv5_2/bn'][1].data.copy()
  w = net.params['inc3a/conv5_2/scale'][0].data.copy()
  c = net.params['inc3a/conv5_2/scale'][1].data.copy()
  alpha = w / np.sqrt(v / n)
  beta = c - (m / n) * alpha
  for i in range(m.shape[0]):
    a[:,i,:,:] = a[:,i,:,:] * alpha[i] + beta[i]
    #a[:,i,:,:] = (a[:,i,:,:] - m[i]) * v[i] * w[i] + c[i]

def combine_conv_bn_scale(net, keyset=None):
  def copy_double(data):
    return np.array(data, copy=True, dtype=np.double)
  if keyset is None:
    keyset = [key[:-3] for key in net.params.keys() if key.endswith('/bn')]

  for key in keyset:
    weight = copy_double(net.params[key][0].data)
    bias = copy_double(net.params[key][1].data)
    num_bn_samples = copy_double(net.params[key + '/bn'][2].data)
    bn_mean = copy_double(net.params[key + '/bn'][0].data)
    bn_variance = copy_double(net.params[key + '/bn'][1].data)
    scale_weight = copy_double(net.params[key + '/scale'][0].data)
    scale_bias = copy_double(net.params[key + '/scale'][1].data)

    if num_bn_samples[0] == 0:
      num_bn_samples[0] = 1
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
    net.params[key][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    for i in range(len(alpha)):
      net.params[key][0].data[i] = weight[i] * alpha[i]

    net.params[key + '/bn'][0].data[:] = 0
    net.params[key + '/bn'][1].data[:] = 1
    net.params[key + '/bn'][2].data[:] = 1
    net.params[key + '/scale'][0].data[:] = 1
    net.params[key + '/scale'][1].data[:] = 0

def combine_conv_bn_scale_pva_inception(net):
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

def load_data(filename):
  f = open(filename, 'rb')
  ndim = unpack("i", f.read(4))[0]
  shape = np.frombuffer(f.read(ndim * 4), dtype=np.int32, count=-1)
  data = np.fromfile(f, dtype=np.float32).reshape(shape)
  f.close()
  return data

def save_data(filename, data):
  f = open(filename, 'wb')
  ndim = len(data.shape)
  f.write(pack('i', ndim))
  f.write(pack('i' * ndim, *data.shape))
  data.tofile(f)
  f.close()

def compress_fc(net, save_dir, layer_name, rank):
  import sys
  sys.path.append('compress')
  import compress
  layer_path = '{:s}/{:s}'.format(save_dir, layer_name)
  W = load_data(layer_path + '_param0.bin')
  b = load_data(layer_path + '_param1.bin')
  W1, b1, W2, b2 = compress.CompressFCLayer(W, b, rank)
  save_data(layer_path + '_L_param0.bin', W1)
  save_data(layer_path + '_L_param1.bin', b1)
  save_data(layer_path + '_U_param0.bin', W2)
  save_data(layer_path + '_U_param1.bin', b2)
  if layer_name + '_L' in net.params.keys():
    net.params[layer_name + '_L'][0].data[:] = W1
    net.params[layer_name + '_L'][1].data[:] = b1
  if layer_name + '_U' in net.params.keys():
    net.params[layer_name + '_U'][0].data[:] = W2
    net.params[layer_name + '_U'][1].data[:] = b2

def convert_net(net, save_dir):
  combine_conv_bn_scale(net)
  for layer_name in net.params.keys():
    if layer_name.endswith('/bn') or layer_name.endswith('/scale'):
      continue
    for param_id in range(len(net.params[layer_name])):
      filename = '{:s}/{:s}_param{:d}.bin'.format(save_dir, layer_name.replace('/', '_'), param_id)
      save_data(filename, net.params[layer_name][param_id].data)
    if layer_name in ['fc6']:
      compress_fc(net, save_dir, layer_name, 512)
    elif layer_name in ['fc7']:
      compress_fc(net, save_dir, layer_name, 128)

def load_image(filename):
  img = scipy.ndimage.imread(filename)

  #if len(img.shape) == 3:
  #  img = img[:,:,:3].swapaxes(0, 2).swapaxes(1, 2)[::-1]
  #elif len(img.shape) == 2:
  #  img = np.tile(img, (3, 1, 1))
  #else:
  #  raise Exception
  #print img.shape

  scale = np.float32(600.0 / min(img.shape[:1]))
  if round(scale * max(img.shape[:1])) > 1000:
    scale = np.float32(1000.0 / max(img.shape[:1]))
  scale_h = np.float32(int(img.shape[0] * scale / 32) * 32.0 / img.shape[0])
  scale_w = np.float32(int(img.shape[1] * scale / 32) * 32.0 / img.shape[1])
  print [scale_h, scale_w]

  im_h = int(round(img.shape[0] * scale_h))
  im_w = int(round(img.shape[1] * scale_w))
  img_scaled = scipy.misc.imresize(img, (im_h, im_w))
  return img_scaled, np.array([im_h, im_w, scale_w, scale_h], np.float32).reshape(1, 4)

if __name__ == "__main__":
  net = load_inception()
  convert_net(net, 'data/temp5')
  net.save('convbnscale.caffemodel')
