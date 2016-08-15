import xml.etree.ElementTree as et

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

def visualize(xml_name, img_path=''):
  import os, cv2
  import matplotlib.pyplot as plt
  folder, filename, objs = parse(xml_name)
  path = os.path.join(img_path, filename)
  print path
  img = cv2.imread(path, 1)
  for label, box in objs:
    cv2.rectangle(img, \
      (box[0], box[1]), (box[2], box[3]), \
      (0, 0, 255), 2)
  cv2.imshow(filename, img)
  key = cv2.waitKey(0)
  cv2.destroyAllWindows()
  #plt.imshow(img[:,:,::-1])
  #plt.show()

if __name__ == '__main__':
  import sys
  visualize(sys.argv[1], img_pah=sys.argv[2])
