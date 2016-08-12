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
        children[key] = [elem.to_dict() \
                         for elem in elems]

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
      folder = self.children['folder'][0].to_dict()
      name = self.children['filename'][0].to_dict()
      size = self.children['size'][0].to_dict()
      kwargs['folder'] = folder
      kwargs['filename'] = name
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
      labels = [elem.to_dict() \
                for elem in self.children['name']]
    else:
      labels = ['__unlabeled__']
    if self.children.has_key('pose'):
      poses = [elem.to_dict() \
               for elem in self.children['pose']]
    else:
      poses = ['Unspecified']
    kwargs['obj_ids'] = [self.id \
                         for _ in labels]
    kwargs['parent_ids'] = [self.id \
                            for _ in labels]
    kwargs['labels'] = labels
    kwargs['poses'] = poses
    super(Object, self).process(**kwargs)

class Part(Element):
  def process(self, **kwargs):
    if self.children.has_key('name'):
      parts = [elem.to_dict() \
               for elem in self.children['name']]
    else:
      parts = ['__unlabeled__']
    if kwargs.has_key('labels') and \
       kwargs.has_key('poses'):
      labels = [label + ' ' + part \
                for label in kwargs['labels'] \
                  for part in parts]
      poses = [pose \
               for pose in kwargs['poses'] \
                 for part in parts]
      kwargs['obj_ids'] = [self.id \
                           for _ in labels]
      kwargs['parent_ids'] = [self.parent.id \
                              for _ in labels]
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
    if kwargs.has_key('folder') and \
       kwargs.has_key('filename') and \
       kwargs.has_key('size') and \
       kwargs['size'] is not None and \
       kwargs.has_key('obj_ids') and \
       kwargs.has_key('parent_ids') and \
       kwargs.has_key('labels') and \
       kwargs.has_key('poses'):
      box = self.to_dict()
      if box is not None:
        folder = kwargs['folder']
        filename = kwargs['filename']
        size = kwargs['size']
        for obj_id, parent_id, label, pose in zip(kwargs['obj_ids'], kwargs['parent_ids'], kwargs['labels'], kwargs['poses']):
          print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % \
              (folder, filename, size[0], size[1], \
               obj_id, parent_id, label, pose, box[0], box[1], box[2], box[3])
    super(BndBox, self).process(**kwargs)

class ElementFactory(object):
  elem_class = { \
    'annotation': Image, \
    'size': Size, \
    'object': Object, \
    'part': Part, \
    'bndbox': BndBox, \
  }

  def __init__(self):
    self.num_special_elems = 0

  def new_elem(self, tag):
    if not ElementFactory.elem_class.has_key(tag):
      return Element(tag, 0)
    self.num_special_elems += 1
    id = self.num_special_elems
    return ElementFactory.elem_class[tag](tag, id)

elem_factory = ElementFactory()

class XMLHandler(object):
  def __init__(self):
    self.init()

  def init(self):
    self.elem = elem_factory.new_elem('(root)')

  def start(self, tag, attributes):
    new_elem = elem_factory.new_elem(tag)
    self.elem.add_child(new_elem)
    self.elem = new_elem

  def end(self, tag):
    self.elem = self.elem.parent

  def data(self, content):
    content = content.strip()
    if len(content) > 0:
      self.elem.add_value(content)

  def comment(self, text):
    #print '[COMMENT] %s' % text
    pass

  def close(self):
    #print 'file closed'
    self.elem.process()
    self.init()

if __name__ == '__main__':
  parser = etree.XMLParser(target = XMLHandler())

  import os, sys
  root_dir = sys.argv[1]
  subs = os.walk(os.path.expanduser(root_dir))
  filenames = [os.path.join(path, fname) \
               for path, subdir, fnames in subs \
                 for fname in fnames]
  filenames.sort()
  for filename in filenames:
    if filename[-4:] == '.xml':
      filepath = os.path.join(root_dir, filename)
      xml_text = open(filepath, 'r').read()
      etree.XML(xml_text, parser)
