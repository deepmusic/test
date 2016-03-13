from struct import pack, unpack
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc

def load_data(filename):
  f = open(filename, 'rb')
  ndim = unpack("i", f.read(4))[0]
  shape = np.frombuffer(f.read(ndim * 4), dtype=np.int32, count=-1)
  data = np.fromfile(f, dtype=np.float32).reshape(shape)
  f.close()
  return data

def save_data(filename, data):
  f = open(filename, 'wb')
  ndim = len(data.shape)
  f.write(pack('i', ndim))
  f.write(pack('i' * ndim, *data.shape))
  data.tofile(f)
  f.close()

def load_image(filename):
  img = scipy.ndimage.imread(filename)

  #if len(img.shape) == 3:
  #  img = img[:,:,:3].swapaxes(0, 2).swapaxes(1, 2)[::-1]
  #elif len(img.shape) == 2:
  #  img = np.tile(img, (3, 1, 1))
  #else:
  #  raise Exception
  #print img.shape

  scale = np.float32(600.0 / min(img.shape[:1]))
  if round(scale * max(img.shape[:1])) > 1000:
    scale = np.float32(1000.0 / max(img.shape[:1]))
  scale_h = np.float32(int(img.shape[0] * scale / 32) * 32.0 / img.shape[0])
  scale_w = np.float32(int(img.shape[1] * scale / 32) * 32.0 / img.shape[1])
  print [scale_h, scale_w]

  im_h = int(round(img.shape[0] * scale_h))
  im_w = int(round(img.shape[1] * scale_w))
  img_scaled = scipy.misc.imresize(img, (im_h, im_w))
  return img_scaled, np.array([im_h, im_w, scale_w, scale_h], np.float32).reshape(1, 4)

def layer_img(img, layer):
  scale_h = img.shape[0] / layer.shape[1]
  scale_w = img.shape[1] / layer.shape[2]
  patch_height = 91
  patch_width = 91
  layer_flat = layer.reshape((layer.shape[0], -1))
  mean_patches = np.zeros((layer.shape[0], patch_height, patch_width, 3), dtype=np.uint8)
  mean_patch = np.zeros((patch_height, patch_width, 3), dtype=np.double)
  weight = np.zeros((1,), dtype=np.double)
  for ch in range(layer.shape[0]):
    weight[0] = 0
    for i, val in enumerate(layer_flat[ch]):
      if val > 0:
            x_ctr = (i % layer.shape[2]) * scale_w
            y_ctr = (i / layer.shape[2]) * scale_h
            x1 = min(max(0, x_ctr), img.shape[1])
            y1 = min(max(0, y_ctr), img.shape[0])
            x2 = min(max(0, x_ctr + patch_width), img.shape[1])
            y2 = min(max(0, y_ctr + patch_height), img.shape[0])
            w = x2 - x1
            h = y2 - y1
            mean_patch[:h, :w, :] += val * img[y1:y2, x1:x2, :]
            weight[0] += val
    mean_patches[ch] = mean_patch / weight[0]
  return mean_patches

def layer_topk(img, layer, ch, num_top=64):
  scale_h = img.shape[0] / layer.shape[1]
  scale_w = img.shape[1] / layer.shape[2]
  patch_height = 91
  patch_width = 91
  layer_flat = layer.reshape((layer.shape[0], -1))
  val = np.ceil(np.sqrt(num_top))
  topk = np.argsort(layer_flat[ch])[::-1][:num_top]
  scores = layer_flat[ch, topk]
  for i, idx in enumerate(topk):
      x_ctr = (idx % layer.shape[2]) * scale_w
      y_ctr = (idx / layer.shape[2]) * scale_h
      x1 = min(max(0, x_ctr), img.shape[1])
      y1 = min(max(0, y_ctr), img.shape[0])
      x2 = min(max(0, x_ctr + patch_width), img.shape[1])
      y2 = min(max(0, y_ctr + patch_height), img.shape[0])
      print '{:d}: ({:f},{:f}), ({:f},{:f}), score = {:f}'.format(idx, x1, y1, x2, y2, scores[i])
      w = x2 - x1
      h = y2 - y1
      plt.subplot(val, val, i + 1)
      img_sub = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
      img_sub[:h, :w, :] = img[y1:y2, x1:x2, :]
      if h < patch_height:
        img_sub[h:, :, :] = 0
      if w < patch_width:
        img_sub[:, w:, :] = 0
      fig = plt.imshow(img_sub)
      plt.axis=('off')
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)
  plt.show()

def plot_imgs(imgs):
  val = np.ceil(np.sqrt(imgs.shape[0]))
  for i in range(imgs.shape[0]):
    plt.subplot(val, val, i + 1)
    fig = plt.imshow(imgs[i], interpolation='bilinear')
    plt.axis=('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
  plt.show()

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

img, im_info = load_image('data/voc/2007/VOC2007/JPEGImages/000004.jpg')
a = load_data('data/temp/conv_bottom0.bin')
#b = layer_img(img, a[0])
#plot_imgs(b)
