import ctypes
from scipy.ndimage import imread
import numpy as np

lib = ctypes.CDLL('libdlcpu.so')

lib._batch_size_net.restype = ctypes.c_int
lib._max_num_bottoms.restype = ctypes.c_int
lib._max_num_tops.restype = ctypes.c_int
lib._max_num_params.restype = ctypes.c_int
lib._max_num_auxs.restype = ctypes.c_int
lib._max_num_ops_per_layer.restype = ctypes.c_int

batch_size = lib._batch_size_net()
max_num_bottoms = lib._max_num_bottoms()
max_num_tops = lib._max_num_tops()
max_num_params = lib._max_num_params()
max_num_auxs = lib._max_num_auxs()
max_num_ops_per_layer = lib._max_num_ops_per_layer()

class Tensor(ctypes.Structure):
  _fields_ = [('name', ctypes.c_char * 32),
              ('num_items', ctypes.c_int),
              ('ndim', ctypes.c_int),
              ('shape', (ctypes.c_int * 5) * batch_size),
              ('start', ctypes.c_int * batch_size),
              ('data', ctypes.POINTER(ctypes.c_float)),
              ('alive_until', ctypes.c_void_p),
              ('has_own_memory', ctypes.c_int),
              ('data_id', ctypes.c_int),
              ('max_data_size', ctypes.c_long)]

class LayerOption(ctypes.Structure):
  _fields_ = [('num_groups', ctypes.c_int),
              ('out_channels', ctypes.c_int),
              ('kernel_h', ctypes.c_int),
              ('kernel_w', ctypes.c_int),
              ('pad_h', ctypes.c_int),
              ('pad_w', ctypes.c_int),
              ('stride_h', ctypes.c_int),
              ('stride_w', ctypes.c_int),
              ('bias', ctypes.c_int),
              ('handle', ctypes.c_void_p),
              ('pooled_height', ctypes.c_int),
              ('pooled_width', ctypes.c_int),
              ('spatial_scale', ctypes.c_float),
              ('flatten', ctypes.c_int),
              ('negative_slope', ctypes.c_float),
              ('scales', ctypes.POINTER(ctypes.c_float)),
              ('ratios', ctypes.POINTER(ctypes.c_float)),
              ('num_scales', ctypes.c_int),
              ('num_ratios', ctypes.c_int),
              ('num_concats', ctypes.c_int),
              ('base_size', ctypes.c_int),
              ('feat_stride', ctypes.c_int),
              ('min_size', ctypes.c_int),
              ('pre_nms_topn', ctypes.c_int),
              ('post_nms_topn', ctypes.c_int),
              ('nms_thresh', ctypes.c_float),
              ('score_thresh', ctypes.c_float),
              ('scaled', ctypes.c_int),
              ('test', ctypes.c_int),
              ('threshold', ctypes.c_float),
              ('scale_weight', ctypes.c_float),
              ('scale_bias', ctypes.c_float),
              ('num_bottoms', ctypes.c_int)]

class Layer(ctypes.Structure):
  _fields_ = [('name', ctypes.c_char * 32),
              ('p_bottoms', ctypes.POINTER(Tensor) * max_num_bottoms),
              ('num_bottoms', ctypes.c_int),
              ('tops', ctypes.POINTER(Tensor) * max_num_tops),
              ('num_tops', ctypes.c_int),
              ('params', ctypes.POINTER(Tensor) * max_num_params),
              ('num_params', ctypes.c_int),
              ('p_aux_data', ctypes.POINTER(ctypes.c_float) * max_num_auxs),
              ('num_aux_data', ctypes.c_int),
              ('f_forward', ctypes.c_void_p * max_num_ops_per_layer),
              ('f_shape', ctypes.c_void_p * max_num_ops_per_layer),
              ('f_init', ctypes.c_void_p * max_num_ops_per_layer),
              ('option', LayerOption)]

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

def top_data(layer_id, top_id = 0):
  try:
    top = lib._layer_net(layer_id, top_id).contents
    shape = np.ctypeslib.as_array(top.shape)[0, :top.ndim]
    size = np.prod(shape)
    p_data = ctypes.addressof(top.data.contents)
    data = np.ctypeslib.as_array((ctypes.c_float * size).from_address(p_data)).reshape(shape)
    return data
  except Exception:
    return None

def logging(layer_id):
  lib._print_layer(layer_id)
