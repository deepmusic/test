from numpy.random import permutation

def to_img_name(name, id):
  return '%s/%s_%04d.jpg' % (name, name, int(id))

def get_label_pair(line):
  tokens = line.strip().split('\t')
  if len(tokens) == 3:
    name = tokens[0]
    id1 = tokens[1]
    id2 = tokens[2]
    return (to_img_name(name, id1), to_img_name(name, id2), 1)
  if len(tokens) == 4:
    name1 = tokens[0]
    id1 = tokens[1]
    name2 = tokens[2]
    id2 = tokens[3]
    return (to_img_name(name1, id1), to_img_name(name2, id2), 0)
  return None

def get_label_sets(filename, shuffled=False):
  set1 = []
  set2 = []
  for line in open(filename, 'r'):
    pair = get_label_pair(line)
    if pair is None:
      continue
    img_name1 = pair[0]
    img_name2 = pair[1]
    sim_label = pair[2]
    set1.append([img_name1, sim_label])
    set2.append([img_name2, sim_label])
  if shuffled:
    shuffled_idx = permutation(len(set1))
    set1 = [set1[i] for i in shuffled_idx]
    set2 = [set2[i] for i in shuffled_idx]
  return (set1, set2)

def make_lmdb_label_sets(src_file, dst1_file, dst2_file, shuffled=False):
  set1, set2 = get_label_sets(src_file, shuffled=shuffled)
  f = open(dst1_file, 'w')
  for img_name, label in set1:
    f.write('%s %d\n' % (img_name, label))
  f.close()
  f = open(dst2_file, 'w')
  for img_name, label in set2:
    f.write('%s %d\n' % (img_name, label))
  f.close()

make_lmdb_label_sets('pairsDevTrain.txt', 'train_1.txt', 'train_2.txt', shuffled=True)
make_lmdb_label_sets('pairsDevTest.txt', 'test_1.txt', 'test_2.txt')
