from caffe.proto.caffe_pb2 import Datum
from lmdb import open as lmdb_open
from PIL.Image import open as image_open
from StringIO import StringIO
import numpy as np 

def db_open(db_dir):
  cursor = lmdb_open(db_dir, readonly=True).begin().cursor()
  return cursor

def db_close(cursor):
  cursor.close()

def db_read(cursor):
  datum = Datum()
  for key, value in cursor:
    datum.ParseFromString(value)
    buf = StringIO()
    buf.write(datum.data)
    buf.seek(0)

    data = np.array(image_open(buf))
    data = data[:, :, ::-1]
    data = np.rollaxis(data, 2, 0)
    yield (key, data, datum.label)

def test(db_dir):
  import cv2
  cursor = db_open(db_dir)
  for i, (key, data, label) in enumerate(db_read(cursor)):
    print 'Datum {:d}: {:s}, {:s}, {:d}'.format(i, key, data.shape, label)
    a = cv2.imread(key[9:])
    a = np.rollaxis(cv2.resize(a, (256,256)), 2, 0)
    print '{:s} {:s}'.format(a[:,95,128], data[:,95,128])
  db_close(cursor)

test('test_db')
