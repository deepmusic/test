import caffe
import argparse
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Non-zero shift bug fix for PVA-9.0.0 model')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel_src',
                        help='original PVA-9.0.0 model',
                        default=None, type=str)
    parser.add_argument('--dest', dest='caffemodel_dest',
                        help='fixed model filename',
                        default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def fix_shift(net):
  for name in net._layer_names:
    if name.endswith('/neg'):
      key = name[:-4]
      channels = net.params[key + '/conv'][0].shape[0]
      net.params[key + '/scale'][1].data[channels:] += net.params[key + '/scale'][0].data[channels:]
      print 'Fix {:s}/scale[{:d}:{:d}]'.format(key, channels, sum(net.params[key + '/scale'][1].data.shape))
    if name.endswith('/input'):
      key = name[:-6]
      if net.params.has_key(key + '/3/bn_scale'):
        net.params[key + '/3/bn_scale'][1].data[:] += 1
        print 'Fix {:s}/3/bn_scale[0:{:d}]'.format(key, sum(net.params[key + '/3/bn_scale'][1].data.shape))
      elif net.params.has_key(key + '/out/bn_scale'):
        net.params[key + '/out/bn_scale'][1].data[:] += 1
        print 'Fix {:s}/out/bn_scale[0:{:d}]'.format(key, sum(net.params[key + '/out/bn_scale'][1].data.shape))
      else:
        print '[Error] Cannot find the scale layer for {:s}'.format(key)

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  net = caffe.Net(args.prototxt, args.caffemodel_src, caffe.TEST)
  fix_shift(net)
  net.save(args.caffemodel_dest)
