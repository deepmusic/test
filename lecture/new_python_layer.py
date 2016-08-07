import caffe
import numpy as np
import yaml

class NewPythonLayer(caffe.Layer):
  def setup(self, bottom, top):
    # Read & parse parameters
    # You can make any optional auxiliary data
    layer_params = yaml.load(self.param_str)
    self._num_output = layer_params['num_output']
    # Compute & set parameter data shape
    self.blobs.add_blob(1,2,3)
    self.blobs.add_blob(1,2,3)
    # Initialize parameter data
    self.blobs[0].data[...] = 0
    self.blobs[1].data[...] = 0
  def reshape(self, bottom, top):
    # Read input data shape
    bottom0_shape = bottom[0].data.shape
    bottom1_shape = bottom[1].data.shape
    # Compute & set output data shape
    top[0].reshape(bottom0_shape[0],bottom0_shape[1],bottom1_shape[2],bottom1_shape[3])
    top[1].reshape(bottom1_shape[0],bottom1_shape[1],bottom1_shape[2],bottom1_shape[3])
  def forward(self, bottom, top):
    # Read input data
    bottom0_data = bottom[0].data
    bottom1_data = bottom[1].data
    # Read parameter data
    weight = self.blobs[0].data
    bias = self.blobs[1].data
    # Compute & store output data
    top[0].data[...] = 0
    top[1].data[...] = 0
  def backward(self, top, propagate_down, bottom):
    # Read output data gradient
    top0_diff = top[0].diff
    top1_diff = top[1].diff
    # Compute & store parameter data gradient
    self.blobs[0].diff[...] = 0
    self.blobs[1].diff[...] = 0
    # If propagate down,
    # compute & store input data gradient as well
    if propagate_down[0]:
      bottom[0].diff[...] = 0
    if propagate_down[1]:
      bottom[1].diff[...] = 0
