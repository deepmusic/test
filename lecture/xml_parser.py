from lxml import etree

class Element(object):
  def __init__(self, tag, id):
    self.tag = tag
    self.id = str(id)
    self.parent = None
    self.children = {}
    self.value = []

  def add_child(self, elem):
    key = elem.tag
    if not self.children.has_key(key):
      self.children[key] = []
    self.children[key].append(elem)
    elem.parent = self

  def add_value(self, value):
    self.value.append(value)

  def to_dict(self):
    children = {}
    for key, elems in self.children.iteritems():
      if len(elems) == 0:
        children.pop(key)
      elif len(elems) == 1:
        children[key] = elems[0].to_dict()
      else:
        children[key] = [elem.to_dict() for elem in elems]

    if len(self.value) == 0:
      value = None
    elif len(self.value) == 1:
      value = self.value[0]
    else:
      value = self.value

    if len(children) > 0:
      if value is not None:
        return (value, children)
      else:
        return children
    else:
      if value is not None:
        return value
      return None

  def to_string(self):
    return str(self.to_dict())

  def process(self, **kwargs):
    for key, elems in self.children.iteritems():
      for elem in elems:
        elem.process(**kwargs)

class Image(Element):
  def process(self, **kwargs):
    if self.children.has_key('folder') and \
       self.children.has_key('filename') and \
       self.children.has_key('size'):
      name = self.children['folder'][0].to_dict() \
             + '/' + self.children['filename'][0].to_dict()
      size = self.children['size'][0].to_dict()
      kwargs['file_path'] = name
      kwargs['size'] = size
    super(Image, self).process(**kwargs)

class Size(Element):
  def to_dict(self):
    try:
      size = [ \
        self.children['height'][0].to_dict(), \
        self.children['width'][0].to_dict(), \
      ]
      return size
    except Exception:
      return None

class Object(Element):
  def process(self, **kwargs):
    if self.children.has_key('name'):
      labels = [elem.to_dict() for elem in self.children['name']]
    else:
      labels = ['__unlabeled__']
    if self.children.has_key('pose'):
      poses = [elem.to_dict() for elem in self.children['pose']]
    else:
      poses = ['Unspecified']
    kwargs['obj_ids'] = [self.id for label in labels]
    kwargs['parent_ids'] = [self.id for label in labels]
    kwargs['labels'] = labels
    kwargs['poses'] = poses
    super(Object, self).process(**kwargs)

class Part(Element):
  def process(self, **kwargs):
    if self.children.has_key('name'):
      parts = [elem.to_dict() for elem in self.children['name']]
    else:
      parts = ['__unlabeled__']
    if kwargs.has_key('labels') and kwargs.has_key('poses'):
      labels = [ \
        label + ' ' + part \
        for label, pose in zip(kwargs['labels'], kwargs['poses']) \
          for part in parts \
      ]
      poses = [pose for pose in kwargs['poses'] for part in parts]
      kwargs['obj_ids'] = [self.id for label in labels]
      kwargs['parent_ids'] = [self.parent.id for label in labels]
      kwargs['labels'] = labels
      kwargs['poses'] = poses
    super(Part, self).process(**kwargs)

class BndBox(Element):
  def to_dict(self):
    try:
      bndbox = [ \
        self.children['xmin'][0].to_dict(), \
        self.children['ymin'][0].to_dict(), \
        self.children['xmax'][0].to_dict(), \
        self.children['ymax'][0].to_dict(), \
      ]
      return bndbox
    except Exception:
      return None

  def process(self, **kwargs):
    if kwargs.has_key('file_path') and \
       kwargs.has_key('size') and kwargs['size'] is not None and \
       kwargs.has_key('obj_ids') and \
       kwargs.has_key('parent_ids') and \
       kwargs.has_key('labels') and \
       kwargs.has_key('poses'):
      box = self.to_dict()
      if box is not None:
        file_path = kwargs['file_path']
        size = kwargs['size']
        for obj_id, parent_id, label, pose in zip(kwargs['obj_ids'], kwargs['parent_ids'], kwargs['labels'], kwargs['poses']):
          print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % \
              (file_path, size[0], size[1], \
               obj_id, parent_id, label, pose, box[0], box[1], box[2], box[3])
    super(BndBox, self).process(**kwargs)

class ElementFactory(object):
  elem_class_map = { \
    'annotation': Image, \
    'size': Size, \
    'object': Object, \
    'part': Part, \
    'bndbox': BndBox, \
  }

  def __init__(self):
    self.mapped_elem_count = 0

  def new_elem(self, tag):
    if ElementFactory.elem_class_map.has_key(tag):
      self.mapped_elem_count += 1
      return ElementFactory.elem_class_map[tag](tag, self.mapped_elem_count)
    return Element(tag, 0)

elem_factory = ElementFactory()

class XMLHandler(object):
  def __init__(self):
    self.init()

  def init(self):
    self.current_elem = elem_factory.new_elem('(root)')

  def start(self, tag, attributes):
    new_elem = elem_factory.new_elem(tag)
    self.current_elem.add_child(new_elem)
    self.current_elem = new_elem

  def end(self, tag):
    self.current_elem = self.current_elem.parent

  def data(self, content):
    content = content.strip()
    if len(content) > 0:
      self.current_elem.add_value(content)

  def comment(self, text):
    #print '[COMMENT] %s' % text
    pass

  def close(self):
    #print 'file closed'
    self.current_elem.process()
    self.init()

parser = etree.XMLParser(target = XMLHandler())

import os, sys
root_dir = sys.argv[1]
filenames = [ \
  os.path.join(path, fname) \
    for path, subdir, fnames in os.walk(os.path.expanduser(root_dir)) \
      for fname in fnames \
]
filenames.sort()
for filename in filenames:
  if filename[-4:] == '.xml':
    xml_text = open(os.path.join(root_dir, filename), 'r').read()
    etree.XML(xml_text, parser)
