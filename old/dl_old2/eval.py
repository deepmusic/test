from datasets.factory import get_imdb
from fast_rcnn.nms_wrapper import nms
import heapq
from struct import unpack
from os.path import join
import numpy as np
import sys

def load_detections(filename, all_boxes):
  f = open(filename, 'rb')
  try:
    while True:
      box = unpack('iifffff', f.read(28))
      print box
      all_boxes[box[1]][box[0]].append(box[2:])
  except Exception:
    print 'loaded'
  f.close()

def eval(filename, **kwargs):
  if kwargs is not None:
    dbname = kwargs['dbname'] if kwargs.has_key('dbname') else 'pvtdb:voc2007test:20'
    outdir = kwargs['outdir'] if kwargs.has_key('outdir') else '../test/data/temp'
  imdb = get_imdb(dbname)
  all_boxes = [[[] for _ in xrange(len(imdb.image_index))] \
               for _ in xrange(imdb.num_classes)]
  load_detections(filename, all_boxes)
  for j in xrange(imdb.num_classes):
    for i in xrange(len(imdb.image_index)):
      all_boxes[j][i] = np.array(all_boxes[j][i], dtype=np.float32)
  nms_dets = apply_nms(all_boxes, 0.3)
  imdb.evaluate_detections(nms_dets, outdir)

def get_list(filename, **kwargs):
  if kwargs is not None:
    dbname = kwargs['dbname'] if kwargs.has_key('dbname') else 'pvtdb:voc2007test:20'
    base_dir = kwargs['base_dir'] if kwargs.has_key('base_dir') else '../data/voc/2007/VOC2007/JPEGImages'
  imdb = get_imdb(dbname)
  f = open(filename, 'w')
  for elem in imdb.image_index:
    f.write(join(base_dir, elem[1]) + '.jpg\n')
  f.close()

def apply_nms(all_boxes, thresh):
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(num_classes)]
  for cls_ind in xrange(num_classes):
    for im_ind in xrange(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue
      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(filename, **kwargs):
  if kwargs is not None:
    dbname = kwargs['dbname'] if kwargs.has_key('dbname') else 'pvtdb:voc2007test:20'
    outdir = kwargs['outdir'] if kwargs.has_key('outdir') else '../test/data/temp'
  imdb = get_imdb(dbname)

  num_images = len(imdb.image_index)
  max_per_set = 40 * num_images
  max_per_image = 100
  thresh = -np.inf * np.ones(imdb.num_classes)
  top_scores = [[] for _ in xrange(imdb.num_classes)]
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  f = open(filename, 'rb')
  for i in xrange(num_images):
    ndim = unpack("i", f.read(4))[0]
    shape = np.frombuffer(f.read(ndim * 4), dtype=np.int32, count=-1)
    num_boxes = shape[0] / (imdb.num_classes - 1)
    data = np.frombuffer(f.read(np.prod(shape) * 4), dtype=np.float32, count=-1) \
             .reshape((imdb.num_classes-1, num_boxes, 6))
    scores = data[:,:,5].swapaxes(0, 1)
    boxes = data[:,:,1:5].swapaxes(0, 1).reshape(num_boxes, -1)
    print [i, scores.shape, boxes.shape]

    for j in xrange(1, imdb.num_classes):
      inds = np.where(scores[:, j-1] > thresh[j])[0]
      cls_scores = scores[inds, j-1]
      cls_boxes = boxes[inds, (j-1)*4:j*4]
      top_inds = np.argsort(-cls_scores)[:max_per_image]
      cls_scores = cls_scores[top_inds]
      cls_boxes = cls_boxes[top_inds, :]
      for val in cls_scores:
        heapq.heappush(top_scores[j], val)
      if len(top_scores[j]) > max_per_set:
        while len(top_scores[j]) > max_per_set:
          heapq.heappop(top_scores[j])
        thresh[j] = top_scores[j][0]

      all_boxes[j][i] = \
           np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
           .astype(np.float32, copy=False)
  f.close()
  for j in xrange(1, imdb.num_classes):
    for i in xrange(num_images):
      inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
      all_boxes[j][i] = all_boxes[j][i][inds, :]

  print 'Applying NMS to all detections'
  nms_dets = apply_nms(all_boxes, 0.3)

  print 'Evaluating detections'
  imdb.evaluate_detections(nms_dets, 'data/temp')

if __name__ == '__main__':
  test_net(sys.argv[1])
