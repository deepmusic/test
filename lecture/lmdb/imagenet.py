import lmdb
from caffe.proto import caffe_pb2
import numpy as np
import cv2

def write_lmdb(db_path, list_filename, height, width, count_start=0, encode=False):
  map_size = 2 * 50000 * 256 * 256 * 3
  db = lmdb.open(db_path, map_size=map_size)
  writer = db.begin(write=True)
  datum = caffe_pb2.Datum()
  for count, line in enumerate(open(list_filename, 'r')):
    img_filename, label = line.strip().split(' ')
    datum.label = int(label)
    datum.channels = 3
    datum.height = height
    datum.width = width
    if encode:
      datum.encoded = True
      img = cv2.resize(cv2.imread(img_filename, 1), (height, width))
      _, img_jpg = cv2.imencode('.jpg', img)
      datum.data = img_jpg.tostring()
    else:
      datum.data = np.rollaxis(img, 2, 0).tostring()
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
    if encoded:
      img = cv2.imdecode(np_array, 1)
      data = np.rollaxis(img, 2, 0)
    else:
      data = np_array.reshape(datum.channels, datum.height, datum.width)
    label = datum.label
    print (key, label, data.shape, data[2,16,16])
    #yield (data, label)
  cursor.close()
  db.close()

def compare_lmdb(db_path1, db_path2, encoded1=False, encoded2=False):
  db1 = lmdb.open(db_path1, readonly=True)
  db2 = lmdb.open(db_path2, readonly=True)
  cursor1 = db1.begin().cursor()
  cursor2 = db2.begin().cursor()
  datum = caffe_pb2.Datum()
  for (key1, val1), (key2, val2) in zip(cursor1, cursor2):
    datum.ParseFromString(val1)
    np_array = np.fromstring(datum.data, dtype=np.uint8)
    if encoded1:
      img = cv2.imdecode(np_array, 1)
      data1 = np.rollaxis(img, 2, 0)
    else:
      data1 = np_array.reshape(datum.channels, datum.height, datum.width)
    label1 = datum.label
    datum.ParseFromString(val2)
    np_array = np.fromstring(datum.data, dtype=np.uint8)
    if encoded2:
      img = cv2.imdecode(np_array, 1)
      data2 = np.rollaxis(img, 2, 0)
    else:
      data2 = np_array.reshape(datum.channels, datum.height, datum.width)
    label2 = datum.label
    if (data1 != data2).sum() > 0:
      print '%d %d %d' % (label1, label2, (data1 != data2).sum())
  cursor1.close()
  cursor2.close()
  db1.close()
  db2.close()

compare_lmdb('ilsvrc12_val_lmdb1', 'ilsvrc12_val_lmdb2', encoded1=True, encoded2=True)

#read_lmdb('ilsvrc12_val_lmdb', encoded=True)
#read_lmdb('val_lmdb', encoded=True)

#for (data1, label1), (data2, label2) in zip(read_lmdb('ilsvrc12_val_lmdb', encoded=True), read_lmdb('val_lmdb', encoded=True)):
#  print '%d %d %d' % (label1, label2, (data1 != data2).sum())

#sum_BGR = np.zeros((3,), dtype=np.double)
#for count, (data, label) in enumerate(read_lmdb('cifar10_train_lmdb', encoded=True)):
#  sum_BGR = sum_BGR * (count / (count + 1.0)) + data.reshape((data.shape[0], -1)).mean(axis=1) * (1 / (count + 1.0))
#  print sum_BGR
