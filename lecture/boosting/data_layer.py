import caffe
import numpy as np
from json import loads

class WeightedSamplingDataLayer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2
      raise Exception("Need two inputs (index, weight) to update sampling distribution")
    layer_params = loads(self.param_str_)
    self._num_data = layer_params['num_data']
    self._weight = np.ones((self._num_data,), dtype=np.float32) / self._num_data
    self._initial_pass = True

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    if not self._initial_pass:
      self._update_sampling_dist(bottom[0], bottom[1])
    data, label, index = self._get_minibatch()
    top[0].data[...] = data
    top[1].data[...] = label
    top[2].data[...] = index

  def backward(self, top, propagate_down, bottom):
    pass

  def _update_sampling_dist(self, index, weight):
    self._weight[index] *= weight
    self._weight /= self._weight.sum()

  def _get_minibatch():
    
