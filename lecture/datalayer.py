import caffe
from caffe.proto.caffe_pb2 import Datum
from lmdb import open as lmdb_open
import numpy as np
import cv2

class DataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self.txn = lmdb_open('lmdb/test_db', readonly=True).begin()
    self.cursor = self.txn.cursor()
    self.cursor.next()
    self.datum = Datum()

  def reshape(self, bottom, top):
    self.datum.ParseFromString(self.cursor.value())
    img_jpg = np.fromstring(self.datum.data, dtype=np.uint8)
    img = cv2.imdecode(img_jpg, 1)
    data = np.tile(np.rollaxis(img, 2, 0), (1, 1, 1, 1))
    top[0].reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    if len(top) == 2:
      top[1].reshape(1, 1)

  def forward(self, bottom, top):
    self.datum.ParseFromString(self.cursor.value())
    img_jpg = np.fromstring(self.datum.data, dtype=np.uint8)
    img = cv2.imdecode(img_jpg, 1)
    data = np.tile(np.rollaxis(img, 2, 0), (1, 1, 1, 1))
    top[0].data[...] = data
    if len(top) == 2:
      top[1].data[...] = self.datum.label
      print (data.shape, self.datum.label)
    if not self.cursor.next():
      self.cursor = self.txn.cursor()
      self.cursor.next()

  def backward(self, top, propagate_down, bottom):
    pass
