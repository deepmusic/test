from caffe import NetSpec, layers, params
from caffe.proto import caffe_pb2

def param(layer_name, param_id, lr_mult=1, decay_mult=1):
  param = caffe_pb2.ParamSpec()
  param.name = '{:s}_param{:d}'.format(layer_name, param_id)
  param.lr_mult = lr_mult
  param.decay_mult = decay_mult
  return param

def param_weight_bias(layer_name, scale_mult=1, bias_term=True):
  weight = param(layer_name, 0, lr_mult=scale_mult, decay_mult=scale_mult)
  if bias_term:
    bias = param(layer_name, 1, lr_mult=2*scale_mult, decay_mult=0)
    return [weight, bias]
  else:
    return [weight]

def empty_layer(layer_name, type_name, bottom_names, top_names, param_list=None):
  layer = caffe_pb2.LayerParameter()
  layer.name = layer_name
  layer.type = type_name
  layer.bottom.extend(bottom_names)
  layer.top.extend(top_names)
  if param_list is not None:
    layer.param.extend(param_list)
  return layer

def convolution_layer(bottom, top_name, num_output, kernel_size, stride=1, pad=None, group=1, bias_term=True, param_layer=None, scale_mult=1):
  layer_name = top_name + '/conv'
  if param_layer is None:
    param_list = param_weight_bias(layer_name, scale_mult=scale_mult, bias_term=bias_term)
  else:
    param_list = param_layer.param
  layer = empty_layer(layer_name, 'Convolution', bottom.top, [top_name], param_list)
  layer.convolution_param.num_output = num_output
  layer.convolution_param.kernel_size = kernel_size
  layer.convolution_param.stride = stride
  layer.convolution_param.pad = pad if pad is not None else (kernel_size - 1) / 2
  layer.convolution_param.group = group
  layer.convolution_param.bias_term = bias_term and len(param_list) > 1
  layer.convolution_param.weight_filler.type = 'xavier'
  if layer.convolution_param.bias_term:
    layer.bias_filler.type = 'constant'
    layer.convolution_param.bias_filler.value = 0
  return layer

def convolution_module(bottom, top_name, num_output, kernel_size, stride=1, pad=None, group=1, param_layer=[None, None, None, None], scale_mult=1):
  conv = convolution_layer(bottom, top_name, num_output, kernel_size, stride, pad, group, False, param_layer[0], scale_mult)
  bn = empty_layer(top_name + '/bn', 'BatcnNorm', [top_name], [top_name], [param(top_name + '/bn', 0, 0, 0) for i in range(3)])
  param_list = param_layer[2].param if param_layer[2] is not None else param_weight_bias(top_name + '/scale', scale_mult, True)
  scale = empty_layer(top_name + '/scale', 'Scale', [top_name], [top_name], param_list)
  scale.scale_param.bias_term = True and len(param_list) > 1
  relu = empty_layer(top_name + '/relu', 'ReLU', [top_name], [top_name])
  return [conv, bn, scale, relu]

def conv_relu_bn_scale(bottom, top_name, num_output, kernel_size, stride=1, pad=None, group=1):
  if pad is None:
    pad = (kernel_size - 1) / 2
  top = layers.Convolution(bottom, name=top_name, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, group=group)
  top = layers.ReLU(top, name=top_name, in_place=True)
  return top

def max_pool(bottom, kernel_size, stride=1):
  return layers.Pooling(bottom, pool=params.Pooling.MAX, kernel_size=kernel_size, stride=stride)

def ave_pool(bottom, kernel_size, stride=1, global_pooling=False):
  return layers.Pooling(bottom, pool=params.Pooling.AVE, kernel_size=kernel_size, stride=stride, global_pooling=global_pooling)

#net = NetSpec()
#net.data, net.label = layers.DataLayer(ntop=2)
#net.conv1 = conv_relu_bn_scale(net.data, 'conv1', 32, 3, 2)

net1 = caffe_pb2.NetParameter()
layer1 = empty_layer('l1', 'Convolution', ['b1', 'b2', 'b3'], ['t1'], param_weight_bias('l1', 0.1, False))
net1.layer.extend([layer1])
layer2 = convolution_module(layer1, 'conv1', 32, 3)
net1.layer.extend(layer2)
layer3 = convolution_module(layer2[0], 'conv2', 32, 3, param_layer=layer2)
net1.layer.extend(layer3)

f = open('myproto.pt', 'w')
#f.write(str(net.to_proto()))
f.write(str(net1))
