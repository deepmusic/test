from proto import caffe_pb2
from google.protobuf import text_format

name = '../../new-faster-rcnn/pva_inception2_coco.pt'

net = caffe_pb2.NetParameter()
f = open(name, 'r')
text_format.Merge(f.read(), net)
f.close()

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
