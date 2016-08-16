from caffe import Layer
import numpy as np

def iou(x, y):
  Ax = (x[:,0] + x[:,1]) * (x[:,2] + x[:,3])
  Ay = (y[:,0] + y[:,1]) * (y[:,2] + y[:,3])
  I_w = np.minimum(x[:,0], y[:,0]) + \
        np.minimum(x[:,1], y[:,1])
  I_h = np.minimum(x[:,2], y[:,2]) + \
        np.minimum(x[:,3], y[:,3])
  AI = I_w * I_h
  AU = Ax + Ay - AI
  return (AI, AU)

class ODLossLayer(Layer):
  def setup(self, bottom, top):
    # bottom[0]: predicted box at each pixel
    #            x = (xl, xr, xt, xb)
    # bottom[1]: regression target at each pixel
    #            y = (yl, yr, yt, yb, bg)
    #            bg: 1 if pixel is in background
    #            (pixel of bg = 1 is ignored)
    # top[0]: loss = ln IoU(x, y)
    top[0].reshape(1)

  def forward(self, bottom, top):
    X = bottom[0].data
    Y = bottom[1].data
    loss = 0
    for x, y in zip(X, Y):
      is_fg = y[:,4] != 1
      x = x[is_fg, :]
      y = y[is_fg, :]
      AI, AU = iou(x, y)
      loss -= np.log(AI / AU).sum()
    top[0].data[...] = loss

  def backward(self, top, propagate_down, bottom):
    Y = bottom[1].data
    X = np.rollaxis(bottom[0].data, 1, 4)
    batch_size, h, w, _ = X.shape
    num_p = h * w
    X = X.reshape(batch_size, num_p, 4)
    diff = np.zeros((batch_size, num_p, 4), \
                    dtype=np.float32)
    for x, y in zip(X, Y):
      is_fg = y[:,4] != 1
      x = x[is_fg, :]
      y = y[is_fg, :]
      AI, AU = iou(x, y)
      I_h = np.minimum(x[:,2], y[:,2]) + \
            np.minimum(x[:,3], y[:,3])
      dAI = (x[:,0] < y[:,0]) * I_h * \
            (AI + AU) / (Ai * AU)
      dAx = (x[:,2] + x[:,3]) / AU
      diff[n, is_fg, :] = -dAI + dAx
    diff = np.rollaxis(diff, 2, 1)
    diff = diff.reshape(batch_size, 4, h, w)
    bottom[0].diff[...] = diff
