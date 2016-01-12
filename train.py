import caffe
import lowrank
import os
import sys
from pva import model_cfg
from ilsvrc2012 import data_info

caffe.set_mode_gpu()
#caffe.set_device(3)

prototxt_postfix = '.prototxt'
caffemodel_postfix = '.caffemodel'

layers = model_cfg['layers']
dump_path = model_cfg['dump_path']
image_path = data_info['image_path']

def TrueNet(dump_path=dump_path, name=layers[0][0]):
    net_true = caffe.Net(os.path.join(dump_path, name + prototxt_postfix), \
                         os.path.join(dump_path, name + caffemodel_postfix), \
                         caffe.TEST)
    lowrank.DumpParams(dump_path, net_true)
    return net_true

def UpdateNet(proto_new, model, layer, model_new, param_path=dump_path):
    net = caffe.Net(proto_new, model, caffe.TEST)
    lowrank.UpdateNetLayerParams(param_path, layer, net)
    net.save(model_new)

def TrainLayer(net_true, layer_idx, dump_path=dump_path, layers=layers):
    layer_prev = layers[layer_idx-1][0]
    layer_now = layers[layer_idx][0]
    rank = layers[layer_idx][1]

    proto_prev = os.path.join(dump_path, layer_prev + prototxt_postfix)
    model_prev = os.path.join(dump_path, layer_prev + caffemodel_postfix)
    proto_new = os.path.join(dump_path, layer_now + prototxt_postfix)
    model_new = os.path.join(dump_path, layer_now + caffemodel_postfix)

    print '########### Learning Layer: %s ###########' % layer_now
    net = caffe.Net(proto_prev, model_prev, caffe.TEST)
    lowrank.Run(dump_path, layer_now, rank, net_true, net, image_path)
    UpdateNet(proto_new, model_prev, layer_now, model_new)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        layer_idx = int(sys.argv[1])
    else:
        layer_idx = 1

    net_true = TrueNet()
    for i in range(layer_idx, len(layers)):
        TrainLayer(net_true, i)
