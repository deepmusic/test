import caffe
import lowrank
import os
import sys
import config

caffe.set_mode_gpu()
#caffe.set_device(3)

def TrueNet(filename=layers[0][0]):
    net_true = caffe.Net(os.path.join(config.meta_path, filename + '.prototxt'), \
                         os.path.join(config.data_path, filename + '.caffemodel'), \
                         caffe.TEST)
    lowrank.DumpParams(config.dump_path, net_true)
    return net_true

def UpdateNet(proto_new, model, layer, model_new):
    net = caffe.Net(proto_new, model, caffe.TEST)
    lowrank.UpdateNetLayerParams(config.dump_path, layer, net)
    net.save(model_new)

def TrainLayer(net_true, layer_idx, layers=layers):
    layer_prev = config.model['layers'][layer_idx-1][0]
    layer_now, rank = config.model['layers'][layer_idx]

    proto_prev = os.path.join(config.meta_path, layer_prev + '.prototxt')
    model_prev = os.path.join(config.data_path, layer_prev + '.caffemodel')
    proto_new = os.path.join(config.meta_path, layer_now + '.prototxt')
    model_new = os.path.join(config.data_path, layer_now + '.caffemodel')

    print '########### Learning Layer: %s ###########' % layer_now
    net = caffe.Net(proto_prev, model_prev, caffe.TEST)
    lowrank.Run(config.dump_path, config.data, layer_now, rank, net_true, net)
    UpdateNet(proto_new, model_prev, layer_now, model_new)

if __name__ == "__main__":
    model_name = 'pva'
    version = '7.0.1'
    data_name = 'imagenet'
    layer_idx = 1

    config.Initialize(model_name, version, data_name)

    net_true = TrueNet()
    for i in range(layer_idx, len(layers)):
        TrainLayer(net_true, i)
