import ctypes
from scipy.ndimage import imread
import numpy as np

lib = ctypes.CDLL('libdlcpu.so')

lib._batch_size.restype = ctypes.c_int
lib._max_ndim.restype = ctypes.c_int
lib._max_name_len.restype = ctypes.c_int

lib._max_num_bottoms.restype = ctypes.c_int
lib._max_num_tops.restype = ctypes.c_int
lib._max_num_params.restype = ctypes.c_int

lib._max_num_tensors.restype = ctypes.c_int
lib._max_num_layers.restype = ctypes.c_int
lib._max_num_shared_blocks.restype = ctypes.c_int

batch_size = lib._batch_size()
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
  _fields_ = [('input_size', ctypes.c_int),
              ('unit_size', ctypes.c_int),
              ('max_image_size', ctypes.c_int),
              ('bias', ctypes.c_int),
              ('group', ctypes.c_int),
              ('num_output', ctypes.c_int),
              ('kernel_h', ctypes.c_int),
              ('kernel_w', ctypes.c_int),
              ('pad_h', ctypes.c_int),
              ('pad_w', ctypes.c_int),
              ('stride_h', ctypes.c_int),
              ('stride_w', ctypes.c_int),
              ('handle', ctypes.c_void_p),
              ('pooled_height', ctypes.c_int),
              ('pooled_width', ctypes.c_int),
              ('spatial_scale', ctypes.c_float),
              ('flatten_shape', ctypes.c_int),
              ('negative_slope', ctypes.c_float),
              ('anchor_scales', ctypes.POINTER(ctypes.c_float)),
              ('anchor_ratios', ctypes.POINTER(ctypes.c_float)),
              ('num_anchor_scales', ctypes.c_int),
              ('num_anchor_ratios', ctypes.c_int),
              ('base_size', ctypes.c_int),
              ('feat_stride', ctypes.c_int),
              ('min_size', ctypes.c_int),
              ('pre_nms_topn', ctypes.c_int),
              ('post_nms_topn', ctypes.c_int),
              ('nms_thresh', ctypes.c_float),
              ('score_thresh', ctypes.c_float),
              ('bbox_vote', ctypes.c_int),
              ('vote_thresh', ctypes.c_float),
              ('scaled_dropout', ctypes.c_int),
              ('test_dropout', ctypes.c_int),
              ('dropout_ratio', ctypes.c_float),
              ('power_weight', ctypes.c_float),
              ('power_bias', ctypes.c_float),
              ('power_order', ctypes.c_float),
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
              ('f_init', ctypes.c_void_p),
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
              ('temp_cpu_data', ctypes.POINTER(ctypes.c_float)),
              ('temp_data', ctypes.POINTER(ctypes.c_float)),
              ('temp_space', ctypes.c_long),
              ('const_data', ctypes.POINTER(ctypes.c_float)),
              ('const_space', ctypes.c_long),
              ('space_cpu', ctypes.c_long),
              ('space', ctypes.c_long),
              ('initialized', ctypes.c_int),
              ('blas_handle', ctypes.c_int),
              ('p_images', ctypes.c_char_p * batch_size),
              ('image_heights', ctypes.c_int * batch_size),
              ('image_widths', ctypes.c_int * batch_size),
              ('num_images', ctypes.c_int)]

lib.create_empty_net.restype = ctypes.POINTER(Net)

lib.create_pvanet.restype = ctypes.POINTER(Net)

lib.get_tensor_by_name.restype = ctypes.POINTER(Tensor)

lib.get_layer_by_name.restype = ctypes.POINTER(Layer)

lib.process_pvanet.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

lib.add_power_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]

lib.add_relu_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float]

lib.add_proposal_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float]

lib.add_roipool_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]

lib.add_dropout_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]

lib.add_odout_layer.argtypes = [ctypes.POINTER(Net), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float]

def detect(net, filename):
  img = imread(filename)
  if img is not None:
    lib.process_pvanet(net, img.tobytes(), img.shape[0], img.shape[1], None, None, None)

def get_tensor_data(tensor):
  try:
    tensor = tensor.contents
    shape = np.ctypeslib.as_array(tensor.shape)[0, :tensor.ndim]
    size = np.prod(shape)
    p_data = ctypes.addressof(tensor.data.contents)
    data = np.ctypeslib.as_array((ctypes.c_float * size).from_address(p_data)).reshape(shape)
    return data
  except Exception:
    return None
