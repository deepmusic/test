import ctypes
from scipy.ndimage import imread

class Tensor(ctypes.Structure):
  _fields_ = [('name', ctypes.c_char * 32),
              ('num_items', ctypes.c_int),
              ('ndim', ctypes.c_int),
              ('shape', (ctypes.c_int * 5) * 128),
              ('start', ctypes.c_int * 128),
              ('data', ctypes.POINTER(ctypes.c_float))]

lib = ctypes.CDLL('libdlcpu.so')
lib._detect_net.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib._layer_net.argtypes = [ctypes.c_int, ctypes.c_int]
lib._layer_net.restype = ctypes.POINTER(Tensor)

def init():
  lib._init_net()

def release():
  lib._release_net()

def detect(filename):
  img = imread(filename)
  if img is not None:
    lib._detect_net(img.tobytes(), img.shape[1], img.shape[0])

def layer(layer_id, top_id):
  return lib._layer_net(layer_id, top_id)

def logging(layer_id):
  lib._print_layer(layer_id)
