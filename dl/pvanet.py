import ctypes
from scipy.ndimage import imread

lib = ctypes.CDLL('libdlgpu.so')

def init():
  lib._init_net()

def release():
  lib._release_net()

def detect(filename):
  img = imread(filename)
  if img is not None:
    lib._detect_net(img.tobytes(), img.shape[1], img.shape[0])
