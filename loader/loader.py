from proto import caffe_pb2
from google.protobuf import text_format
import ctypes

lib = ctypes.CDLL('libdlcpu.so')
name = '../pvanet/9.0.0_mod1_tuned/9.0.0_mod1.orig.pt'

net = caffe_pb2.NetParameter()
f = open(name, 'r')
text_format.Merge(f.read(), net)
f.close()

def test():
  print net.name
  for i, layer in enumerate(net.layer):
    print 'Layer {:d}: {:s}, {:s}, {:s}'.format(i, layer.name, layer.type, layer.top)
    if layer.convolution_param:
      print 'stride = {:f}'.format(layer.convolution_param.stride)
    if layer.param:
      try:
        print 'lr_mult = {:f}'.format(layer.param[0].lr_mult)
        print 'decay_mult = {:f}'.format(layer.param[0].decay_mult)
        print 'lr_mult = {:f}'.format(layer.param[1].lr_mult)
        print 'decay_mult = {:f}'.format(layer.param[1].decay_mult)
      except:
        None
    if layer.include:
      try:
        if layer.include[0].phase:
          print 'phase = train'
        else:
          print 'phase = test'
      except:
        None

def parse_layer(layer, phase=1):
  def convert_name(name):
    return name.replace('/', '_')
  def convert_names(names):
    return [convert_name(name) for name in names]

  if len(layer.include) > 0 and all([elem.phase != phase for elem in layer.include]):
    return

  print '{:s} {:d}, {:s}'.format(convert_name(layer.name), len(layer.name), layer.type)
  print '  bottom: {:s}'.format(convert_names(layer.bottom))
  print '  top: {:s}'.format(convert_names(layer.top))
  print '  param: {:s}'.format(convert_names([elem.name for elem in layer.param]))

for layer in net.layer:
  parse_layer(layer)
