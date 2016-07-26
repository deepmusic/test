import caffe

def param(param_name, param_id=0, lr_mult=1, decay_mult=1):
  param = caffe.proto.caffe_pb2.ParamSpec()
  param.name = '{:s}/param{:d}'.format(param_name, param_id)
  param.lr_mult = lr_mult
  param.decay_mult = decay_mult
  return param

def param_weight_bias(param_name, scale_mult=1, bias_term=True):
  weight = param(param_name, param_id=0, lr_mult=scale_mult, decay_mult=scale_mult)
  if bias_term:
    bias = param(param_name, param_id=1, lr_mult=2*scale_mult, decay_mult=0)
    return [weight, bias]
  else:
    return [weight]

def empty_layer(layer_name, type_name, bottom_names, top_names, phase=None):
  layer = caffe.proto.caffe_pb2.LayerParameter()
  layer.name = layer_name
  layer.type = type_name
  layer.bottom.extend(bottom_names)
  layer.top.extend(top_names)
  if phase is not None:
    include_param = caffe.proto.caffe_pb2.NetStateRule()
    if phase == 'TRAIN':
      include_param.phase = 0
    elif phase == 'TEST':
      include_param.phase = 1
    layer.include.extend([include_param])
  return layer

def data_layer(source, batch_size=25, top_names=None, crop_size=227, mean_values=None, phase=None):
  layer_name = 'input-data'
  if top_names is None:
    top_names = ['data', 'label']
  if mean_values is None:
    mean_values = [104, 117, 123]
  layer = empty_layer(layer_name, 'Data', [], top_names, phase=phase)
  layer.transform_param.crop_size = crop_size
  layer.transform_param.mean_value.extend(mean_values)
  layer.data_param.source = source
  layer.data_param.batch_size = batch_size
  layer.data_param.backend = caffe.params.Data.LMDB
  return layer

def dummy_data_layer(top_names=None, shapes=None, phase=None):
  layer_name = 'input-data'
  if phase is None:
    phase = 'TEST'
  if top_names is None:
    top_names = ['data', 'label']
  if shapes is None:
    shapes = [[1, 3, 224, 224], [1, 3]]
  layer = empty_layer(layer_name, 'DummyData', [], top_names, phase=phase)
  for shp in shapes:
    shape = caffe.proto.caffe_pb2.BlobShape()
    shape.dim.extend(shp)
    layer.dummy_data_param.shape.extend([shape])
  return layer

def convolution_layer(bottom_name, top_name, num_output, kernel_size, stride=1, pad=None, group=1, bias_term=True, param_ref_name=None, scale_mult=1, phase=None):
  layer_name = top_name + '/conv'
  layer = empty_layer(layer_name, 'Convolution', [bottom_name], [top_name], phase=phase)
  layer.convolution_param.num_output = num_output
  layer.convolution_param.kernel_h = kernel_size
  layer.convolution_param.kernel_w = kernel_size
  layer.convolution_param.stride_h = stride
  layer.convolution_param.stride_w = stride
  layer.convolution_param.pad_h = pad if pad is not None else (kernel_size - 1) / 2
  layer.convolution_param.pad_w = pad if pad is not None else (kernel_size - 1) / 2
  layer.convolution_param.group = group
  layer.convolution_param.bias_term = bias_term
  layer.convolution_param.weight_filler.type = 'xavier'
  if param_ref_name is None:
    param_name = layer_name
  else:
    param_name = param_ref_name + '/conv'
  layer.param.extend(param_weight_bias(param_name, scale_mult=scale_mult, bias_term=bias_term))
  return layer

def pooling_layer(bottom_name, top_name, kernel_size, stride, pool='MAX', phase=None):
  layer_name = top_name + '/pool'
  layer = empty_layer(layer_name, 'Pooling', [bottom_name], [top_name], phase=None)
  layer.pooling_param.kernel_size = kernel_size
  layer.pooling_param.stride = stride
  if pool == 'MAX':
    layer.pooling_param.pool = caffe.params.Pooling.MAX
  elif pool == 'AVE':
    layer.pooling_param.pool = caffe.params.Pooling.AVE
  return layer

def global_pooling_layer(bottom_name, top_name, pool='MAX', phase=None):
  layer_name = top_name + '/pool'
  layer = empty_layer(layer_name, 'Pooling', [bottom_name], [top_name], phase=None)
  layer.pooling_param.global_pooling = True
  if pool == 'MAX':
    layer.pooling_param.pool = caffe.params.Pooling.MAX
  elif pool == 'AVE':
    layer.pooling_param.pool = caffe.params.Pooling.AVE
  return layer

def dropout_layer(bottom_name, top_name, dropout_ratio, phase=None):
  layer_name = top_name + '/dropout'
  layer = empty_layer(layer_name, 'Dropout', [bottom_name], [top_name], phase=None)
  layer.dropout_param.dropout_ratio = dropout_ratio
  return layer

def batch_norm_layer(bottom_name, top_name, param_ref_name=None, phase=None):
  layer_name = top_name + '/bn'
  layer = empty_layer(layer_name, 'BatchNorm', [bottom_name], [top_name], phase=None)
  if param_ref_name is None:
    param_name = layer_name
  else:
    param_name = param_ref_name + '/bn'
  layer.param.extend([param(param_name, param_id=i, lr_mult=0, decay_mult=0) for i in range(3)])
  return layer

def scale_layer(bottom_name, top_name, bias_term=True, param_ref_name=None, scale_mult=1, phase=None):
  layer_name = top_name + '/scale'
  layer = empty_layer(layer_name, 'Scale', [bottom_name], [top_name], phase=None)
  layer.scale_param.bias_term = bias_term
  if param_ref_name is None:
    param_name = layer_name
  else:
    param_name = param_ref_name + '/scale'
  layer.param.extend(param_weight_bias(param_name, scale_mult=scale_mult, bias_term=bias_term))
  return layer

def relu_layer(bottom_name, top_name, phase=None):
  layer_name = top_name + '/relu'
  layer = empty_layer(layer_name, 'ReLU', [bottom_name], [top_name], phase=None)
  return layer

def concat_layer(bottom_names, top_name, axis=1, phase=None):
  layer_name = top_name + '/concat'
  layer = empty_layer(layer_name, 'Concat', bottom_names, [top_name], phase=None)
  layer.concat_param.axis = axis
  return layer

def softmax_loss_layer(bottom_name, top_name, label_name=None, phase=None):
  layer_name = top_name + '/softmax_loss'
  if label_name is None:
    label_name = 'label'
  layer = empty_layer(layer_name, 'SoftmaxWithLoss', [bottom_name, label_name], [top_name], phase=None)
  return layer

def contrastive_loss_layer(bottom1_name, bottom2_name, top_name, label_name=None, margin=1, phase=None):
  layer_name = top_name + '/contrastive_loss'
  if label_name is None:
    label_name = 'label'
  layer = empty_layer(layer_name, 'ContrastiveLoss', [bottom1_name, bottom2_name, label_name], [top_name], phase=None)
  layer.contrastive_loss_param.margin = margin
  return layer

def accuracy_layer(bottom_name, top_name, label_name=None, top_k=1, phase=None):
  layer_name = top_name + '/accuracy'
  if label_name is None:
    label_name = 'label'
  layer = empty_layer(layer_name, 'Accuracy', [bottom_name, label_name], [top_name], phase=None)
  layer.accuracy_param.top_k = top_k
  return layer

def append(lst, elem):
  if elem.__class__ == list:
    lst.extend(elem)
  else:
    lst.extend([elem])

def conv_relu_module(bottom_name, top_name, num_output, kernel_size, stride=1, pad=None, group=1, param_ref_name=None, scale_mult=1):
  module = []
  append(module, convolution_layer(bottom_name, top_name, num_output, kernel_size, stride=stride, pad=pad, group=group, bias_term=True, param_ref_name=param_ref_name, scale_mult=scale_mult))
  append(module, relu_layer(top_name, top_name))
  return module

def conv_scale_relu_module(bottom_name, top_name, num_output, kernel_size, stride=1, pad=None, group=1, param_ref_name=None, scale_mult=1):
  module = []
  append(module, convolution_layer(bottom_name, top_name, num_output, kernel_size, stride=stride, pad=pad, group=group, bias_term=False, param_ref_name=param_ref_name, scale_mult=scale_mult))
  append(module, scale_layer(top_name, top_name, bias_term=True, param_ref_prefix=param_ref_prefix, scale_mult=scale_mult))
  append(module, relu_layer(top_name, top_name))
  return module

def conv_bn_scale_relu_module(bottom_name, top_name, num_output, kernel_size, stride=1, pad=None, group=1, param_ref_name=None, scale_mult=1):
  module = []
  append(module, convolution_layer(bottom_name, top_name, num_output, kernel_size, stride=stride, pad=pad, group=group, bias_term=False, param_ref_name=param_ref_name, scale_mult=scale_mult))
  append(module, batch_norm_layer(top_name, top_name, param_ref_name=param_ref_name))
  append(module, scale_layer(top_name, top_name, bias_term=True, param_ref_name=param_ref_name, scale_mult=scale_mult))
  append(module, relu_layer(top_name, top_name))
  return module

def squeeze_module(conv_module, bottom_name, top_name, num_squeeze, num_expand, param_ref_name=None, scale_mult=1):
  squeeze_name = top_name + '/squeeze1x1'
  expand1x1_name = top_name + '/expand1x1'
  expand3x3_name = top_name + '/expand3x3'

  if param_ref_name is None:
    param_ref_name = top_name
  squeeze_ref = param_ref_name + '/squeeze1x1'
  expand1x1_ref = param_ref_name + '/expand1x1'
  expand3x3_ref = param_ref_name + '/expand3x3'

  module = []
  append(module, conv_module(bottom_name, squeeze_name, num_squeeze, 1, param_ref_name=squeeze_ref, scale_mult=scale_mult))
  append(module, conv_module(squeeze_name, expand1x1_name, num_expand, 1, param_ref_name=expand1x1_ref, scale_mult=scale_mult))
  append(module, conv_module(squeeze_name, expand3x3_name, num_expand, 3, param_ref_name=expand3x3_ref, scale_mult=scale_mult))
  append(module, concat_layer([expand1x1_name, expand3x3_name], top_name))
  return module

def squeeze_net(net=None, name="", param_ref_name="", scale_mult=1):
  if net is None:
    net = caffe.proto.caffe_pb2.NetParameter()
  conv_module = conv_bn_scale_relu_module
  append(net.layer, conv_module('data', name+'conv1', 64, 3, stride=2, pad=0, param_ref_name=param_ref_name+'conv1', scale_mult=scale_mult))
  append(net.layer, pooling_layer(name+'conv1', name+'pool1', 3, 2, pool='MAX'))
  append(net.layer, squeeze_module(conv_module, name+'pool1', name+'fire2', 16, 64, param_ref_name=param_ref_name+'fire2', scale_mult=scale_mult))
  append(net.layer, squeeze_module(conv_module, name+'fire2', name+'fire3', 16, 64, param_ref_name=param_ref_name+'fire3', scale_mult=scale_mult))
  append(net.layer, pooling_layer(name+'fire3', name+'pool3', 3, 2, pool='MAX'))
  append(net.layer, squeeze_module(conv_module, name+'pool3', name+'fire4', 32, 128, param_ref_name=param_ref_name+'fire4', scale_mult=scale_mult))
  append(net.layer, squeeze_module(conv_module, name+'fire4', name+'fire5', 32, 128, param_ref_name=param_ref_name+'fire5', scale_mult=scale_mult))
  append(net.layer, pooling_layer(name+'fire5', name+'pool5', 3, 2, pool='MAX'))
  append(net.layer, squeeze_module(conv_module, name+'pool5', name+'fire6', 48, 192, param_ref_name=param_ref_name+'fire6', scale_mult=scale_mult))
  append(net.layer, squeeze_module(conv_module, name+'fire6', name+'fire7', 48, 192, param_ref_name=param_ref_name+'fire7', scale_mult=scale_mult))
  append(net.layer, squeeze_module(conv_module, name+'fire7', name+'fire8', 64, 256, param_ref_name=param_ref_name+'fire8', scale_mult=scale_mult))
  append(net.layer, squeeze_module(conv_module, name+'fire8', name+'fire9', 64, 256, param_ref_name=param_ref_name+'fire9', scale_mult=scale_mult))
  append(net.layer, dropout_layer(name+'fire9', name+'fire9', 0.5))
  append(net.layer, conv_module(name+'fire9', name+'conv10', 1000, 1, param_ref_name=param_ref_name+'conv10', scale_mult=scale_mult))
  append(net.layer, global_pooling_layer(name+'conv10', name+'out', pool='AVE'))
  return net

def classifier():
  net = caffe.proto.caffe_pb2.NetParameter()
  append(net.layer, data_layer('data/imagenet/ilsvrc12_train_lmdb', batch_size=64, phase='TRAIN'))
  append(net.layer, data_layer('data/imagenet/ilsvrc12_val_lmdb', phase='TEST'))
  squeeze_net(net=net)
  append(net.layer, accuracy_layer('out', 'accuracy', phase='TEST'))
  append(net.layer, accuracy_layer('out', 'accuracy_top5', top_k=5, phase='TEST'))
  return net

def siamese_net():
  #append(net.layer, dummy_data_layer())
  net = caffe.proto.caffe_pb2.NetParameter()
  append(net.layer, data_layer('data/imagenet/ilsvrc12_train_lmdb', batch_size=64, phase='TRAIN'))
  append(net.layer, data_layer('data/imagenet/ilsvrc12_val_lmdb', phase='TEST'))
  squeeze_net(net=net, name="")
  squeeze_net(net=net, name="B/", param_ref_name="")
  append(net.layer, contrastive_loss_layer('out', 'B/out', 'loss'))
  return net

def net_to_file(net, pt_filename):
  f = open(pt_filename, 'w')
  f.write(str(net))
  f.close()

def copy_squeeze_net(src, dst):
  for key in dst._layer_names:
    if key.endswith('/conv'):
      src_key = key[:-5]
      if not src.params.has_key(src_key):
        continue
      print key
      weight = src.params[src_key][0].data
      bias = src.params[src_key][1].data
      scale_key = key[:-5] + '/scale'
      if not dst.params.has_key(scale_key):
        dst.params[key][0].data[:] = weight
        dst.params[key][1].data[:] = bias
      else:
        print scale_key
        weight_shape = weight.shape
        weight = weight.reshape(weight.shape[0], -1)
        weight_norm = (weight ** 2).sum(axis=1) ** 0.5
        normalized_weight = weight / weight_norm.reshape(weight.shape[0], 1)
        dst.params[key][0].data[:] = normalized_weight.reshape(weight_shape)
        dst.params[scale_key][0].data[:] = weight_norm
        dst.params[scale_key][1].data[:] = bias
    elif key.endswith('/bn'):
      print key
      dst.params[key][1].data[:] = 1
      dst.params[key][2].data[:] = 1

def save_squeeze_net(pt_src, model_src, pt_dst, model_dst):
  src = caffe.Net(pt_src, model_src, caffe.TEST)
  dst = caffe.Net(pt_dst, caffe.TEST)
  copy_squeeze_net(src, dst)
  dst.save(model_dst)

net = classifier()
net_to_file(net, 'myproto.pt')
save_squeeze_net('SqueezeNet/SqueezeNet_v1.1/train_val.prototxt', 'SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel', 'myproto.pt', 'mymodel.cm')
