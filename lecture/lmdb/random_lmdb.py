from caffe.proto.caffe_pb2 import Datum
from lmdb import open as lmdb_open
from PIL.Image import open as image_open
from StringIO import StringIO
import numpy as np 
import time as timelib

def db_read_datum(cursor):
  datum = Datum()
  datum.ParseFromString(cursor.value())
  buf = StringIO()
  buf.write(datum.data)
  buf.seek(0)
  data = np.array(image_open(buf))
  data = data[:, :, ::-1]
  data = np.rollaxis(data, 2, 0)
  return (cursor.key(), data, datum.label)

def db_read(reader, index, batch_size):
  cursor = reader.cursor()
  cursor.set_range('%08d' % index)
  batch_key = []
  batch_data = np.zeros((batch_size, 3, 256, 256), dtype=np.float32)
  batch_label = np.zeros((batch_size,), dtype=np.float32)
  time = 0
  for i in range(batch_size):
    start = timelib.time()
    key, data, label = db_read_datum(cursor)
    batch_key.append(key)
    batch_data[i, ...] = data
    batch_label[i, ...] = label
    time = time + timelib.time() - start
    cursor.next()
  cursor.close()
  return (batch_key, batch_data, batch_label, time)

import cv2
reader = lmdb_open('../data/train_lmdb', readonly=True).begin()
mean_time = 0
mean_time2 = 0
for i in range(100):
  index = np.random.randint(1220000)
  start = timelib.time()
  batch_key, batch_data, batch_label, time = db_read(reader, index, 128)
  time2 = timelib.time() - start
  #print batch_key
  mean_time = mean_time * i / (i + 1) + time / (i + 1)
  mean_time2 = mean_time2 * i / (i + 1) + time2 / (i + 1)
  print [mean_time, mean_time2]
