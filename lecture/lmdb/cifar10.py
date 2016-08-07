import cPickle
def unpickle(filename):
  return cPickle.load(open(filename, 'rb'))

import lmdb
from caffe.proto import caffe_pb2
from PIL import Image
import numpy as np
import cv2

def write_lmdb(db_path, dataset_name, count_start=0, encode=False):
  dataset = unpickle(dataset_name)
  map_size = (count_start + len(dataset['labels'])) * 2 * (3 * 32 * 32 + 128)
  db = lmdb.open(db_path, map_size=map_size)
  writer = db.begin(write=True)
  datum = caffe_pb2.Datum()
  for count, (img, label) in enumerate(zip(dataset['data'], dataset['labels'])):
    datum.label = label
    datum.channels = 3
    datum.height = 32
    datum.width = 32
    if encode:
      datum.encoded = True
      img_rgb = np.rollaxis(img.reshape(3, 32, 32), 0, 3)
      _, img_png = cv2.imencode('.jpg', img_rgb)
      datum.data = img_png.tostring()
    else:
      datum.data = img.tostring()
    key = '%010d' % (count + count_start)
    writer.put(key, datum.SerializeToString(), append=True)
    print (key, label)
  writer.commit()
  db.close()

def read_lmdb(db_path, encoded=False):
  db = lmdb.open(db_path, readonly=True)
  reader = db.begin()
  cursor = reader.cursor()
  datum = caffe_pb2.Datum()
  for key, value in cursor:
    datum.ParseFromString(value)
    np_array = np.fromstring(datum.data, dtype=np.uint8)
    print '%d %d %d' % (datum.height, datum.width, datum.channels)
    if encoded:
      img_bgr = cv2.imdecode(np_array, 1)
      data_bgr = np.rollaxis(img_rgb, 2, 0)
    elif CHW_order:
      data_bgr = np_array.reshape(datum.channels, datum.height, datum.width)[::-1, :, :]
    label = datum.label
    #print (key, label, data_bgr.shape, data_bgr[2,16,16])
    yield (data_bgr, label)
  cursor.close()
  db.close()

#for (data1, label1), (data2, label2) in zip(read_lmdb('../../caffe/examples/cifar10/cifar10_train_lmdb', CHW_order=True), read_lmdb('cifar10_train_lmdb', encoded=True)):
#  print '%d %d %d' % (label1, label2, (data1 != data2).sum())

#sum_BGR = np.zeros((3,), dtype=np.double)
#for count, (data, label) in enumerate(read_lmdb('cifar10_train_lmdb', encoded=True)):
#  sum_BGR = sum_BGR * (count / (count + 1.0)) + data.reshape((data.shape[0], -1)).mean(axis=1) * (1 / (count + 1.0))
#  print sum_BGR
