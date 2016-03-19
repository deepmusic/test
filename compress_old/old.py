import caffe
import lowrank
import config

caffe.set_mode_gpu()
caffe.set_device(0)

def Init():
    true_net = caffe.Net('proto/true.prototxt', 'model/true.caffemodel', caffe.TEST)
    compressed_net = caffe.Net('proto/compressed.prototxt', 'model/true.caffemodel', caffe.TEST)
    return (true_net, compressed_net)

def Run(cfgs):
    true_net, compressed_net = Init()
    for i in range(len(cfgs.layers)):
        layer, rank = cfgs.layers[i]
        print '########### Compressing %s to rank %d ###########' % (layer, rank)
        lowrank.Run(cfgs, layer, rank, true_net, compressed_net)
    compressed_net.save('model/compressed.caffemodel')

if __name__ == "__main__":
    cfgs = config.model_['vggnet']
    Run(cfgs)
