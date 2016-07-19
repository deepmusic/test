import ctypes
from scipy.ndimage import imread
import numpy as np

lib = ctypes.CDLL('libdlcpu.so')

lib._batch_size_net.restype = ctypes.c_int
lib._max_ndim.restype = ctypes.c_int
lib._max_name_len.restype = ctypes.c_int

lib._max_num_bottoms.restype = ctypes.c_int
lib._max_num_tops.restype = ctypes.c_int
lib._max_num_params.restype = ctypes.c_int

lib._max_num_tensors.restype = ctypes.c_int
lib._max_num_layers.restype = ctypes.c_int
lib._max_num_shared_blocks.restype = ctypes.c_int

batch_size = lib._batch_size_net()
max_ndim = lib._max_ndim()
max_name_len = lib._max_name_len()

max_num_bottoms = lib._max_num_bottoms()
max_num_tops = lib._max_num_tops()
max_num_params = lib._max_num_params()

max_num_tensors = lib._max_num_tensors()
max_num_layers = lib._max_num_layers()
max_num_shared_blocks = lib._max_num_shared_blocks()

class Tensor(ctypes.Structure):
  _fields_ = [('name', ctypes.c_char * max_name_len),
              ('num_items', ctypes.c_int),
              ('ndim', ctypes.c_int),
              ('shape', (ctypes.c_int * max_ndim) * batch_size),
              ('start', ctypes.c_int * batch_size),
              ('data', ctypes.POINTER(ctypes.c_float)),
              ('data_type', ctypes.c_int)]

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
              ('base_size', ctypes.c_int),
              ('feat_stride', ctypes.c_int),
              ('min_size', ctypes.c_int),
              ('pre_nms_topn', ctypes.c_int),
              ('post_nms_topn', ctypes.c_int),
              ('nms_thresh', ctypes.c_float),
              ('score_thresh', ctypes.c_float),
              ('bbox_vote', ctypes.c_int),
              ('vote_thresh', ctypes.c_float),
              ('scaled', ctypes.c_int),
              ('test', ctypes.c_int),
              ('threshold', ctypes.c_float),
              ('scale_weight', ctypes.c_float),
              ('scale_bias', ctypes.c_float),
              ('num_bottoms', ctypes.c_int),
              ('channel_axis', ctypes.c_int),
              ('reshape', ctypes.c_int * max_ndim),
              ('reshape_ndim', ctypes.c_int)]

class Layer(ctypes.Structure):
  _fields_ = [('name', ctypes.c_char * max_name_len),
              ('p_bottoms', ctypes.POINTER(Tensor) * max_num_bottoms),
              ('num_bottoms', ctypes.c_int),
              ('p_tops', ctypes.POINTER(Tensor) * max_num_tops),
              ('num_tops', ctypes.c_int),
              ('p_params', ctypes.POINTER(Tensor) * max_num_params),
              ('num_params', ctypes.c_int),
              ('aux_data', ctypes.c_void_p),
              ('f_forward', ctypes.c_void_p),
              ('f_shape', ctypes.c_void_p),
              ('f_free', ctypes.c_void_p),
              ('option', LayerOption)]

class Net(ctypes.Structure):
  _fields_ = [('param_path', ctypes.c_char * 1024),
              ('tensors', Tensor * max_num_tensors),
              ('num_tensors', ctypes.c_int),
              ('layers', Layer * max_num_layers),
              ('num_layers', ctypes.c_int),
              ('p_shared_blocks', ctypes.POINTER(ctypes.c_float) * max_num_shared_blocks),
              ('num_shared_blocks', ctypes.c_int),
              ('num_output_boxes', ctypes.c_int),
              ('temp_data', ctypes.POINTER(ctypes.c_float)),
              ('temp_cpu_data', ctypes.POINTER(ctypes.c_float)),
              ('temp_space', ctypes.c_long),
              ('const_data', ctypes.POINTER(ctypes.c_float)),
              ('const_space', ctypes.c_long),
              ('space_cpu', ctypes.c_long),
              ('space', ctypes.c_long),
              ('initialized', ctypes.c_int),
              ('input_scale', ctypes.c_int),
              ('blas_handle', ctypes.c_int)]

lib._net.restype = ctypes.POINTER(Net)
lib.get_tensor_by_name.restype = ctypes.POINTER(Tensor)
lib.get_layer_by_name.restype = ctypes.POINTER(Layer)
lib._detect_net.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib._layer_net.argtypes = [ctypes.c_int, ctypes.c_int]
lib._layer_net.restype = ctypes.POINTER(Tensor)
lib.add_scale_const_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float]
lib.add_relu_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float]
lib.add_proposal_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float]
lib.add_roipool_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.add_dropout_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
lib.add_odout_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float]

def generate():
  lib._generate_net()

def release():
  lib._release_net()

def detect(filename):
  img = imread(filename)
  if img is not None:
    lib._detect_net(img.tobytes(), img.shape[1], img.shape[0])

def net():
  return lib._net().contents

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
