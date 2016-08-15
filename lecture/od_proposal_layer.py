from caffe import Layer
import numpy as np
import cv2
import xml.etree.ElementTree as et
import json
import os

def iou(Bx, By):
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
  # x = (xl, xr, xt, xb)
  # Bx = (p1_j, p1_i, p2_j, p2_i)
  Bx = np.zeros(len(x), 4)
  Bx[:,0] = pj - x[:,0]
  Bx[:,1] = pi - x[:,2]
  Bx[:,2] = pj + x[:,1]
  Bx[:,3] = pi + x[:,3]
  return Bx

def By2y(pi, pj, By):
  y = np.zeros(len(By), 4)
  y[:,0] = pj - By[:,0]
  y[:,1] = By[:,2] - pj
  y[:,2] = pi - By[:,1]
  y[:,3] = By[:,3] - pi
  return y

def target(pi, pj, By):
  is_in = np.zeros((pi.shape[0], By.shape[0]), \
                   dtype=np.bool)
  for n, box in enumerate(By):
    # p1_j <= pj <= p2_j  and  p1_i <= pi <= p2_i
    is_in[:, n] = (box[0] <= pj) * (pj <= box[2]) * \
                  (box[1] <= pi) * (pi <= box[3])
  return is_in

def process(X, BY):
  batch_size, height, width, _ = X.shape
  num_p = height * width
  ri = np.array(range(height)) + 0.5
  rj = np.array(range(width)) + 0.5
  pj, pi = np.meshgrid(rj, ri)
  pj = pj.reshape(-1)
  pi = pi.reshape(-1)
  X = X.reshape(batch_size, num_p, 4)
  Y = np.array((batch_size, num_p, 4),
               dtype=np.float32)
  label = np.array((batch_size, num_p),
                   dtype=np.float32)
  for n, x in enumerate(X):
    By = BY[BY[:,0] == n, 1:5]
    is_in = target(pi, pj, By)
    Bx = x2Bx(pi, pj, x)
    IOU = iou(Bx, By)
    target = np.argmax(IOU * is_in, axis=1)
    Y[n, ...] = By2y(pi, pj, By[target,:])
    label[n, is_in.any(axis=1)] = 1  # 1 - area((p, pj), center of By) ?
    label[n, (-is_in).all(axis=1)] = 0
  return label, Y

class ProposalLayer(Layer):
  def setup(self, bottom, top):
    # bottom[0]: predicted box for each pixel
    #            x = (xl, xr, xt, xb)
    # bottom[1]: ground-truth boxes By
    #            box = (n, p1_j, p1_i, p2_j, p2_i)
    #            n: n-th batch item
    params = json.loads(self.param_str_)
    print params
    self.scale = params['scale']
    # batch size
    batch_size = bottom[0].shape(0)
    # number of pixels
    num_p = bottom[0].shape(2) * bottom[0].shape(3)
    # target objectness for each pixel
    # (positive = 1, negative = 0)
    top[0].reshape(batch_size, num_p)
    # target bounding-box for each pixel
    # y = (yl, yr, yt, yb)
    top[1].reshape(batch_size, num_p * 4)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    # bottom[0]: (batch_size x 4 x height x width)
    # -> X: (batch_size x height x width x 4)
    X = np.rollaxis(bottom[0].data, 1, 4)
    BY = bottom[2].data
    BY[:,1:5] /= self.scale
    label, Y = process(X, BY)
    top[0].reshape(label.shape)
    top[0].data[...] = label
    top[1].reshape(Y.shape)
    top[1].data[...] = Y

  def backward(self, top, propagate_down, bottom):
    pass
