import mxnet as mx
import numpy as np
import caffe

def conv(prev_top, num_layer, num_filter):
    conv1 = mx.symbol.Convolution(name='conv{:d}_1'.format(num_layer), data=prev_top,
                                  kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=num_filter)
    relu1 = mx.symbol.Activation(name='relu{:d}_1'.format(num_layer), data=conv1, act_type='relu')
    conv2 = mx.symbol.Convolution(name='conv{:d}_2'.format(num_layer), data=relu1,
                                  kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter*2)
    relu2 = mx.symbol.Activation(name='relu{:d}_1'.format(num_layer), data=conv2, act_type='relu')
    return relu2

def fc(prev_top, num_layer, num_hidden, dropout=True):
    fc = mx.symbol.FullyConnected(name='fc{:d}'.format(num_layer), data=prev_top, num_hidden=num_hidden)
    relu = mx.symbol.Activation(name='relu{:d}'.format(num_layer), data=fc, act_type='relu')
    drop = mx.symbol.Dropout(name='drop{:d}'.format(num_layer), data=relu, p=0.5)
    return drop

def get_symbol(num_classes=21):
    # data
    layer = mx.symbol.Variable(name='data')

    # conv 1-5
    layer = conv(layer, 1, 16)
    layer = conv(layer, 2, 32)
    layer = conv(layer, 3, 64); layer3 = layer
    layer = conv(layer, 4, 128); layer4 = layer
    layer = conv(layer, 5, 256); layer5 = layer

    # convf
    layer5_slice = mx.symbol.SliceChannel(data=layer5, num_outputs=512, name='conv5_slice')
    deconv = [mx.symbol.Deconvolution(name='upsample_{:d}'.format(i), data=layer5_slice[i],
                                      kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=1,
                                      no_bias=True) for i in range(512)]
    up = mx.symbol.Concat(*deconv, name='concat')
    down = mx.symbol.Pooling(name='downsample', data=layer3,
                             kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max')
    layer = mx.symbol.Concat(*[down, layer4, up], name='concat')
    layer = mx.symbol.Convolution(name='convf', data=layer,
                                  kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=512)
    layer = mx.symbol.Activation(name='reluf', data=layer, act_type='relu')

    # rpn
    layer = mx.symbol.Convolution(name='rpn_conv1', data=layer,
                                  kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=512)
    layer = mx.symbol.Activation(name='rpn_relu1', data=layer, act_type='relu')
    rpn_score = mx.symbol.Convolution(name='rpn_cls_score', data=layer,
                                      kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=50)
    rpn_score = mx.symbol.Reshape(name='rpn_cls_score_reshape', data=rpn_score,
                                  target_shape=(1, 2, 25*40, 40))
    rpn_score = mx.symbol.SoftmaxActivation(name='rpn_cls_prob', data=rpn_score, type='channel')
    rpn_score = mx.symbol.Reshape(name='rpn_cls_prob_reshape', data=rpn_score,
                                  target_shape=(1, 50, 40, 40))
    rpn_bbox = mx.symbol.Convolution(name='rpn_bbox_pred', data=layer,
                                     kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=100)

    # proposal
    roi = mx.symbol.Proposal(*[rpn_score, rpn_bbox], name='proposal',
                             scales=(4, 8, 16, 24, 32))

    # rcnn
    layer = mx.symbol.ROIPooling(*[layer, roi], name='roi_pool_conv5',
                                 pooled_size=(6, 6), spatial_scale=0.0625)
    layer = mx.symbol.Flatten(data=layer[0])
    layer = fc(layer, 6, 4096)
    layer = fc(layer, 7, 4096)
    rcnn_score = mx.symbol.FullyConnected(name='cls_score', data=layer, num_hidden=num_classes)
    rcnn_score = mx.symbol.SoftmaxOutput(name='cls_prob', data=rcnn_score)
    rcnn_bbox = mx.symbol.FullyConnected(name='bbox_pred', data=layer, num_hidden=num_classes*4)

    layer = mx.symbol.Group([rcnn_score, rcnn_bbox])
    return layer

def load_model(caffeproto, caffemodel, num_classes=21):
    caffe_net = caffe.Net(caffeproto, caffemodel, caffe.TEST)
    symbol = get_symbol(num_classes)
    #arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(**{'data': (1, 3, 640, 640), 'im_info': (1, 4)})
    arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(**{'data': (1, 3, 640, 640)})
    arg_shape_dic = { name: shape for name, shape in zip(symbol.list_arguments(), arg_shapes) }

    arg_params = {}
    is_first_conv = True
    for layer_name in caffe_net.params.keys():
        if layer_name.startswith('data') or layer_name.startswith('im_info'):
            continue

        print 'Loading {} parameters'.format(layer_name)
        
        wmat = caffe_net.params[layer_name][0].data
        if is_first_conv and layer_name.startswith('conv'):
            print 'Swapping BGR -> RGB order'
            wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]
            is_first_conv = False

        if layer_name.startswith('upsample'):
            for i in range(wmat.shape[0]):
                weight_name = layer_name + '_{:d}_weight'.format(i)
                if arg_shape_dic.has_key(weight_name):
                    wmat_i = wmat[i].reshape((1, wmat.shape[1], wmat.shape[2], wmat.shape[3]))
                    print 'Original {}: {}'.format(weight_name, wmat_i.shape)
                    print 'Target {}: {}'.format(weight_name, arg_shape_dic[weight_name])
                    arg_params[weight_name] = mx.nd.zeros(arg_shape_dic[weight_name])
                    arg_params[weight_name][:] = wmat_i

        weight_name = layer_name + '_weight'
        if arg_shape_dic.has_key(weight_name):
            print 'Original {}: {}'.format(weight_name, wmat.shape)
            print 'Target {}: {}'.format(weight_name, arg_shape_dic[weight_name])
            arg_params[weight_name] = mx.nd.zeros(arg_shape_dic[weight_name])
            arg_params[weight_name][:] = wmat

        if len(caffe_net.params[layer_name]) == 1:
            continue

        bias = caffe_net.params[layer_name][1].data
        bias_name = layer_name + '_bias'
        if arg_shape_dic.has_key(bias_name):
            print 'Original {}: {}'.format(bias_name, bias.shape)
            print 'Target {}: {}'.format(bias_name, arg_shape_dic[bias_name])
            arg_params[bias_name] = mx.nd.zeros(bias.shape)
            arg_params[bias_name][:] = bias

    model = mx.model.FeedForward(ctx=mx.gpu(), symbol=symbol, arg_params=arg_params,
                                 aux_params={}, num_epoch=1,
                                 learning_rate=0.01, momentum=0.9, wd=0.0001)
    model.save(caffemodel + '.mx')

proto_path = '/home/kye-hyeon/Work/test/meta/pva/7.0.1/full/faster_rcnn_test_anchor25.pt'
model_path = '/home/kye-hyeon/Work/test/data/pva/7.0.1/full/frcnn_anchor25_iter_146000.caffemodel'
load_model(proto_path, model_path)
