from datasets.factory import get_imdb
from struct import unpack
from os.path import join
import numpy as np

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

def get_list(filename, **kwargs):
  if kwargs is not None:
    dbname = kwargs['dbname'] if kwargs.has_key('dbname') else 'pvtdb:voc2007test:20'
    base_dir = kwargs['base_dir'] if kwargs.has_key('base_dir') else '../data/voc/2007/VOC2007/JPEGImages'
  imdb = get_imdb(dbname)
  f = open(filename, 'w')
  for elem in imdb.image_index:
    f.write(join(base_dir, elem[1]) + '.jpg\n')
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
  imdb.evaluate_detections(all_boxes, outdir)
