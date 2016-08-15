from caffe import Layer
import numpy as np
import cv2
import xml.etree.ElementTree as et
import json
import os

def iou_2p(Bx, By):
  Ax = (Bx[:,2] - Bx[:,0]) * (Bx[:,3] - Bx[:,1])
  Ay = (By[:,2] - By[:,0]) * (By[:,3] - By[:,1])
  AU = Ax + Ay
  IOU = np.zeros((Bx.shape[0], By.shape[0]))
  for n, box in enumerate(By):
    J1 = np.maximum(Bx[:,0], box[0])
    I1 = np.maximum(Bx[:,1], box[1])
    J2 = np.minimum(Bx[:,2], box[2])
    I2 = np.minimum(Bx[:,3], box[3])
    AI = np.maximum(I2 - I1, 0) * np.maximum(J2 - J1, 0)
    IOU[:, n] = AI / (AU - AI)
  return IOU

def x2Bx(pi, pj, x):
  Bx[:,0] = pj - x[:,0]
  Bx[:,1] = pj + x[:,1]
  Bx[:,2] = pi - x[:,2]
  Bx[:,3] = pi + x[:,3]
  return Bx

def target(B, pi, pj):
  for n, box in enumerate(B):
    is_in[:, n] = (box[0] <= pj) * (pj <= box[2]) \
                  (box[1] <= pi) * (pi <= box[3])
  return is_in

class ProposalLayer(Layer):
  def setup(self, bottom, top):
    params = json.loads(self.param_str_)
    print params
    self.scale = params['scale']
    self.pi = np.array(range(bottom[0].shape(2))) + 0.5
    self.pj = np.array(range(bottom[0].shape(3))) + 0.5
    num_p = bottom[0].shape(2) * bottom[0].shape(3)
    top[0].reshape(1, num_p)
    top[1].reshape(1, num_p * 4)
    top[2].reshape(1, num_p * 4)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    assert(bottom[0].shape(0) == 1)
    assert(bottom[1].shape(0) == 1)
    x = bottom[1].data[n].reshape(4, -1)

    line = self.source.readline()
    fdir, fname, objs = parse(line.strip())
    if self.use_folder:
      path = os.path.join(self.img_dir, fdir, fname)
    else:
      path = os.path.join(self.img_dir, fname)
    print path
    img = cv2.imread(path)

    short_side = min(img.shape[0], img.shape[1])
    long_side = max(img.shape[0], img.shape[1])
    scale = self.short_min / short_side
    if round(scale * long_side) > self.long_max:
      scale = long_max / long_side
    top[2].reshape(1, 6)
    scale_h = int(img.shape[0] * scale / 32) * 32.0 / img.shape[0]
    scale_w = int(img.shape[1] * scale / 32) * 32.0 / img.shape[1]
    h = int(round(img.shape[0] * scale_h))
    w = int(round(img.shape[1] * scale_w))
    im_info = [h, w, scale_h, scale_w, img.shape[0], img.shape[1]]
    top[2].data[...] = im_info

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.rollaxis(img, 2, 0)
    img = np.array(img, dtype=np.float32)
    img[0,:,:] -= self.mean[0]
    img[1,:,:] -= self.mean[1]
    img[2,:,:] -= self.mean[2]
    top[0].reshape(1, 3, img.shape[1], img.shape[2])
    top[0].data[...] = img
    print top[0].data.shape
    top[1].reshape(len(objs), 5)
    for i, (label, box) in enumerate(objs):
      top[1].data[i, 0:4] = np.asarray(box, dtype=np.float32)
      top[1].data[i, 4] = label_dict[label]

  def backward(self, top, propagate_down, bottom):
    pass
