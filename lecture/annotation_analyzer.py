class Annotation(object):
  # line: img_path, img_height, img_width, obj_id, parent_id, label, pose, xmin, ymin, xmax, ymax
  field_index_map = { \
    'img_path': 0, \
    'img_height': 1, \
    'img_width': 2, \
    'obj_id': 3, \
    'parent_id': 4, \
    'label': 5, \
    'pose': 6, \
    'xmin': 7, \
    'ymin': 8, \
    'xmax': 9, \
    'ymax': 10, \
  }

  def __init__(self, line):
    self.parse_from_line(line)

  def parse_from_line(self, line):
    tokens = line.split('\t')
    for field, index in Annotation.field_index_map.iteritems():
      setattr(self, field, tokens[index])

  def get(self, field):
    return getattr(self, field)

  def count_value(self, field, value_count_map):
    value = self.get(field)
    if not value_count_map.has_key(value):
      value_count_map[value] = 0
    value_count_map[value] += 1

  def satisfy(self, *args):
    return any([all([self.get(field) == value \
                    for field, value in and_conditions.iteritems()]) \
               for and_conditions in args])

  def is_part(self):
    return self.obj_id != self.parent_id

  def is_crowd(self):
    return self.label.endswith(' crowd')

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
num_satisfies = { condition_name: 0 for condition_name in conditions.keys() }
num_all = 0
num_parts = 0
num_crowds = 0

import sys
for line in open(sys.argv[1]):
  anno = Annotation(line)
  num_all += 1
  num_parts += anno.is_part()
  num_crowds += anno.is_crowd()
  for field, value_count_map in num_categories.iteritems():
    anno.count_value(field, value_count_map)
  for condition_name, condition in conditions.iteritems():
    num_satisfies[condition_name] += anno.satisfy(*condition)

print sys.argv[1]
print '====='
print '# anotations: %d' % num_all
print '# parts: %d' % num_parts
print '# crowds: %d' % num_crowds
condition_names = num_satisfies.keys()
condition_names.sort()
for condition_name in condition_names:
  print '%s: %d' % (condition_name, num_satisfies[condition_name])
print '====='
field_names = num_categories.keys()
field_names.sort()
for field_name in field_names:
  print '%s' % field_name
  value_count_map = num_categories[field_name]
  value_names = value_count_map.keys()
  value_names.sort()
  for value in value_names:
    if value.endswith(' crowd'):
      continue
    print '%s: %d' % (value, value_count_map[value])
  print '====='
