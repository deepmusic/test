import caffe

def conv_relu_bn_scale(bottom, top_name, num_output, kernel_size, stride=1, pad=None, group=1):
  if pad is None:
    pad = (kernel_size - 1) / 2
  top = caffe.layers.Convolution(bottom, name=top_name, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, group=group)
  top = caffe.layers.ReLU(top, name=top_name, in_place=True)
  return top

def max_pool(bottom, kernel_size, stride=1):
  return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.MAX, kernel_size=kernel_size, stride=stride)

def ave_pool(bottom, kernel_size, stride=1, global_pooling=False):
  return caffe.layers.Pooling(bottom, pool=caffe.params.Pooling.AVE, kernel_size=kernel_size, stride=stride, global_pooling=global_pooling)

net = caffe.NetSpec()
net.data, net.label = caffe.layers.DataLayer(ntop=2)
net.conv1 = conv_relu_bn_scale(net.data, 'conv1', 32, 3, 2)
f = open('myproto.pt', 'w')
f.write(str(net.to_proto()))
f.close()
