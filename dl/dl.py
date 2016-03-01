from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def load_data(filename):
  f = open(filename, 'rb')
  ndim = unpack("i", f.read(4))[0]
  shape = np.frombuffer(f.read(ndim * 4), dtype=np.int32, count=-1)
  data = np.fromfile(f, dtype=np.float32).reshape(shape)
  f.close()
  return data

def load_image(filename):
  img = scipy.ndimage.imread(filename)

  if len(img.shape) == 3:
    img = img[:,:,:3].swapaxes(0, 2).swapaxes(1, 2)[::-1]
  elif len(img.shape) == 2:
    img = np.tile(img, (3, 1, 1))
  else:
    raise Exception
  print img.shape

  scale = np.float32(600.0 / min(img.shape[1:]))
  if round(scale * max(img.shape[1:])) > 1000:
    scale = np.float32(1000.0 / max(img.shape[1:]))
  scale_h = np.float32(int(img.shape[1] * scale / 32) * 32.0 / img.shape[1])
  scale_w = np.float32(int(img.shape[2] * scale / 32) * 32.0 / img.shape[2])
  print [scale_h, scale_w]

  im_h = int(round(img.shape[1] * scale_h))
  im_w = int(round(img.shape[2] * scale_w))
  return np.array([im_h, im_w, scale_w, scale_h], np.float32).reshape(1, 4)

def plot_f(weight, bias, swap_bgr=False):
  val = np.ceil(np.sqrt(weight.shape[0]))
  for i in range(weight.shape[0]):
    plt.subplot(val, val, i + 1)
    w = weight[i][::-1] if swap_bgr else weight[i]
    img_shape = np.zeros((len(w.shape) + 1,))
    img_shape[1:] = w.shape
    img_shape[0] = 100000
    imgs = np.random.uniform(0, 255, img_shape)
    scores = [(w * img).sum() + bias[i] for img in imgs]
    simg_mean = np.array([0 * img if score <= 0 else score * img for score, img in zip(scores, imgs)]).mean(axis=0)
    s_mean = np.array([0 if score <= 0 else score for score in scores]).mean()
    print s_mean
    fig = plt.imshow(simg_mean / s_mean, interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
  plt.show()
