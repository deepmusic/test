from caffe import Layer
import numpy as np
import cv2
import xml.etree.ElementTree as et
import json
import os

def parse_value(node, key):
  value = node.find(key).text
  return int(round(float(value)))

def parse_object(node):
  label = ''
  box = []
  for child in node:
    if child.tag == 'name':
      label = child.text
    elif child.tag == 'bndbox':
      box = [parse_value(child, 'xmin'), \
             parse_value(child, 'ymin'), \
             parse_value(child, 'xmax'), \
             parse_value(child, 'ymax')]
  return (label, box)

def parse_root(root):
  folder = ''
  filename = ''
  objs = []
  for child in root:
    if child.tag == 'folder':
      folder = child.text
    elif child.tag == 'filename':
      filename = child.text
    elif child.tag == 'object':
      objs.append(parse_object(child))
  return (folder, filename, objs)

def parse(filename):
  tree = et.parse(filename)
  root = tree.getroot()
  return parse_root(root)

labels = ('__background__', \
  'bicycle', 'bird', 'bus', 'car', 'cat', \
  'dog', 'horse', 'motorbike', 'person', 'train', \
  'aeroplane', 'boat', 'bottle', 'chair', 'cow', \
  'diningtable', 'pottedplant', 'sheep', 'sofa', \
  'tvmonitor')
label_dict = { label: i \
               for i, label in enumerate(labels) }

class ODDataLayer(Layer):
  def setup(self, bottom, top):
    params = json.loads(self.param_str)
    self.source = open(params['source'], 'r')
    self.use_folder = params['use_folder']
    self.img_dir = params['img_dir']
    self.mean = params['mean']
    self.short_min = params['short_min']
    self.long_max = params['long_max']
    top[0].reshape(1, 3, self.short_min / 32 * 32, \
                         self.long_max / 32 * 32)
    top[1].reshape(1, 5)
    if len(top) > 2:
      top[2].reshape(1, 6)
    self.short_min = float(self.short_min)
    self.long_max = float(self.long_max)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    line = self.source.readline()
    fdir, fname, objs = parse(line.strip())
    if self.use_folder:
      path = os.path.join(self.img_dir, fdir, fname)
    else:
      path = os.path.join(self.img_dir, fname)
    #print path
    img = cv2.imread(path)

    short_side = min(img.shape[0], img.shape[1])
    long_side = max(img.shape[0], img.shape[1])
    scale = self.short_min / short_side
    if round(scale * long_side) > self.long_max:
      scale = self.long_max / long_side
    scale_h = int(img.shape[0] * scale / 32) * 32.0 / img.shape[0]
    scale_w = int(img.shape[1] * scale / 32) * 32.0 / img.shape[1]
    h = int(round(img.shape[0] * scale_h))
    w = int(round(img.shape[1] * scale_w))
    if len(top) > 2:
      im_info = [h, w, scale_h, scale_w, img.shape[0], img.shape[1]]
      top[2].reshape(1, 6)
      top[2].data[...] = im_info

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.rollaxis(img, 2, 0)
    img = np.array(img, dtype=np.float32)
    img[0,:,:] -= self.mean[0]
    img[1,:,:] -= self.mean[1]
    img[2,:,:] -= self.mean[2]
    top[0].reshape(1, 3, img.shape[1], img.shape[2])
    top[0].data[...] = img
    #print top[0].data.shape
    top[1].reshape(len(objs), 5)
    for i, (label, box) in enumerate(objs):
      top[1].data[i, 1:5] = np.asarray(box, dtype=np.float32)
      top[1].data[i, 0] = 0
      #top[1].data[i, 0:4] = np.asarray(box, dtype=np.float32)
      #top[1].data[i, 4] = label_dict[label]

  def backward(self, top, propagate_down, bottom):
    pass
