class Annotation(object):
  fields = [ \
    'img_folder', 'img_name', \
    'img_h', 'img_w', \
    'obj_id', 'parent_id', \
    'label', 'pose', \
    'xmin', 'ymin', 'xmax', 'ymax', \
  ]
  fld_idx = dict(zip(fields, range(len(fields))))

  def __init__(self, line):
    self.parse_from_line(line)
    self.img_h = int(self.img_h)
    self.img_w = int(self.img_w)
    self.xmin = int(self.xmin)
    self.ymin = int(self.ymin)
    self.xmax = int(self.xmax)
    self.ymax = int(self.ymax)

  def parse_from_line(self, line):
    tokens = line.split('\t')
    for fld, idx in Annotation.fld_idx.items():
      setattr(self, fld, tokens[idx])

  def get(self, field):
    return getattr(self, field)

class AnalyzableAnnotation(Annotation):
  def count_value(self, field, value_count_map):
    value = self.get(field)
    if not value_count_map.has_key(value):
      value_count_map[value] = 0
    value_count_map[value] += 1

  def satisfy(self, *args):
    return any([all([self.get(field) == value \
        for field, value in conjunctions.items()]) \
          for conjunctions in args])

  def is_part(self):
    return self.obj_id != self.parent_id

  def is_crowd(self):
    return self.label.endswith(' crowd')

import cv2
import numpy as np
class PlotableAnnotation(Annotation):
  def load(self, prefix='', midfix='', postfix=''):
    img_path = prefix + self.img_folder + \
               midfix + self.img_name + postfix
    print img_path
    img = cv2.imread(img_path, 1)
    box_h = self.ymax - self.ymin
    box_w = self.xmax - self.xmin
    y = self.ymin + int(np.ceil(0.5 * box_h))
    x = self.xmin + int(np.ceil(0.5 * box_w))
    crop_size = max(box_h, box_w)
    ct = max(0, crop_size - y)
    cb = max(0, y + crop_size - self.img_h)
    cl = max(0, crop_size - x)
    cr = max(0, x + crop_size - self.img_w)
    print (ct, cb, cl, cr)
    it = max(0, y - crop_size)
    ib = min(self.img_h, y + crop_size)
    il = max(0, x - crop_size)
    ir = min(self.img_w, x + crop_size)
    print (it, ib, il, ir)
    crop = np.zeros((crop_size*2, crop_size*2, 3), \
                    dtype=np.uint8)
    print crop.shape
    box = [crop_size-(x-self.xmin), crop_size-(y-self.ymin), crop_size+x-self.xmin, crop_size+y-self.ymin]
    crop[ct:crop_size*2-cb, cl:crop_size*2-cr, :] = \
        img[it:ib, il:ir, :]
    return crop, box

for line in open('annotation_voc2007.txt', 'r'):
  a = PlotableAnnotation(line)
  img, box = a.load(prefix='/home/kye-hyeon/Work/data/pvtdb/', midfix='/JPEGImages/', postfix='')
  cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
  cv2.imshow(a.img_name, img)
  key = cv2.waitKey(0)
  cv2.destroyAllWindows()
  if key == 27:
    break

if __name__ == '__main__':
  fields = [ 'label', 'pose' ]
  conditions = { \
    '# person': [ \
      { 'label': 'person' }, \
    ], \
    '# person w/o pose': [ \
      { 'label': 'person', 'pose': 'Unspecified' }, \
    ], \
    '# vehicle': [ \
      { 'label': 'car' }, \
      { 'label': 'bus' }, \
      { 'label': 'bicycle' }, \
      { 'label': 'motorbike' }, \
      { 'label': 'truck' }, \
     ], \
    '# vehicle w/o pose': [ \
      { 'label': 'car', 'pose': 'Unspecified' }, \
      { 'label': 'bus', 'pose': 'Unspecified' }, \
      { 'label': 'bicycle', 'pose': 'Unspecified' }, \
      { 'label': 'motorbike', 'pose': 'Unspecified' }, \
      { 'label': 'truck', 'pose': 'Unspecified' }, \
     ], \
    '# animal': [ \
      { 'label': 'dog' }, \
      { 'label': 'cat' }, \
     ], \
    '# animal w/o pose': [ \
      { 'label': 'dog', 'pose': 'Unspecified' }, \
      { 'label': 'cat', 'pose': 'Unspecified' }, \
     ], \
    '# traffic light': [ \
      { 'label': 'traffic light' }, \
     ], \
    '# traffic light w/o pose': [ \
      { 'label': 'traffic light', 'pose': 'Unspecified' }, \
     ], \
  }

  num_categories = { field: {} for field in fields }
  num_satisfies = { name: 0 for name in conditions.keys() }
  num_total_annotations = 0
  num_parts = 0
  num_crowds = 0

  import sys
  for line in open(sys.argv[1]):
    anno = AnalyzableAnnotation(line)
    num_total_annotations += 1
    num_parts += anno.is_part()
    num_crowds += anno.is_crowd()
    for field, val_cnt in num_categories.items():
      anno.count_value(field, val_cnt)
    for name, condition in conditions.items():
      num_satisfies[name] += anno.satisfy(*condition)

  print sys.argv[1]
  print '====='
  print '# anotations: %d' % num_total_annotations
  print '# parts: %d' % num_parts
  print '# crowds: %d' % num_crowds
  condition_names = num_satisfies.keys()
  condition_names.sort()
  for name in condition_names:
    print '%s: %d' % (name, num_satisfies[name])
  print '====='
  field_names = num_categories.keys()
  field_names.sort()
  for name in field_names:
    print '%s' % name
    val_cnt = num_categories[name]
    value_names = val_cnt.keys()
    value_names.sort()
    for value in value_names:
      if value.endswith(' crowd'):
        continue
      print '%s: %d' % (value, val_cnt[value])
    print '====='
