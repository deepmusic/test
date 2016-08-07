import caffe
import numpy as np
import time as timelib
import matplotlib.pyplot as plt

class GraphVisualizer(object):
  def __init__(self):
    self.loss_list = []
    self.legend = []
    self.total_count = 0
    self.time = 0
    self.fig, self.axes = plt.subplots()
    self.fig.show()

  def add_new_loss(self, legend, loss):
    if len(self.loss_list) == 0:
      self.loss_list = [[] for _ in range(len(loss))]
      self.legend = legend
    for i, val in enumerate(loss):
      self.loss_list[i].append(val)
    self.total_count += 1

  def filter_old_loss(self):
    for lst in self.loss_list:
      threshold = np.mean(lst) * 2 - np.min(lst)
      count = 0
      for val in lst:
        if val < threshold:
          break
        count += 1
      del lst[:count]

  def update(self, legend, loss, time):
    self.add_new_loss(legend, loss)
    self.time = time
    if self.total_count % 1000 == 0:
      self.filter_old_loss()
    if self.total_count % 50 == 0:
      self.redraw()

  def redraw(self):
    self.axes.clear()
    axes_range = None
    for name, lst in zip(self.legend, self.loss_list):
      num_loss = len(lst)
      if num_loss < 2:
        continue

      x_data = np.array(np.linspace(0, num_loss-1, min(num_loss, 1000)), dtype=int)
      y_data = [lst[x] for x in x_data]
      x_data += self.total_count - num_loss
      self.axes.plot(x_data, y_data, label=name)

      if axes_range is None:
        axes_range = [self.total_count - num_loss, np.min(y_data), np.max(y_data)]
      else:
        axes_range[0] = min(axes_range[0], self.total_count - num_loss)
        axes_range[1] = min(axes_range[1], np.min(y_data))
        axes_range[2] = max(axes_range[2], np.max(y_data))

    if axes_range is not None:
      self.axes.set_title('Running time = %.3fs/iteration' % self.time)
      self.axes.set_xlim([axes_range[0], self.total_count])
      self.axes.set_ylim([axes_range[1], axes_range[2]])
      legend = self.axes.legend(loc='upper right')
      self.axes.figure.canvas.draw()
      plt.pause(0.01)

def init(solver_name, solverstate_name=None, caffemodel_name=None):
  solver = caffe.SGDSolver(solver_name)
  if solverstate_name is not None:
    solver.restore(solverstate_name)
  elif caffemodel_name is not None:
    solver.net.copy_from(caffemodel_name)
  return solver

def train(solver, visualize=False, average_until=10000, ignored=None):
  mean_factor = None
  mean_loss_list = []

  plateau_stepsize = 20000

  if visualize:
    try:
      graph = GraphVisualizer()
    except RuntimeError:
      visualize = False
  if ignored is None:
    ignored = []
  try:
    while True:
      start = timelib.time()
      solver.step(1)

      legend = [key for key in solver.net.outputs if key not in ignored]
      loss = np.array([solver.net.blobs[key].data for key in legend]).flatten()
      time = timelib.time() - start
      if mean_factor is None:
        mean_factor = 0.1 ** (1.0 / average_until)
        mean_loss = loss
        mean_time = time
      else:
        mean_loss = mean_loss * mean_factor + loss * (1 - mean_factor)
        mean_time = mean_time * mean_factor + time * (1 - mean_factor)
      mean_loss_list.append(mean_loss)
      if len(mean_loss_list) > plateau_stepsize and mean_loss > 0.999 * mean_loss_list[-plateau_stepsize]:
        print 'plateau occurred: %.3f vs. %.3f' % (mean_loss, mean_loss_list[-plateau_stepsize])
        solver.set_base_lr(solver.get_base_lr() * 0.5)
        plateau_stepsize *= 2
        mean_loss_list = []
      if visualize:
        graph.update(legend, mean_loss, mean_time)
      else:
        print 'Loss = %s (current), %s (moving average)' % (str(zip(legend, loss)), str(zip(legend, mean_loss)))
        print 'Time = %.3f (current), %.3f (moving average)' % (time, mean_time)
  finally:
    solver.snapshot()
    if visualize:
      graph.redraw()
      plt.show()

caffe.set_mode_gpu()
caffe.set_device(0)
solver = init('practice2_1_solver.pt')
solver.restore('practice2_1_train_iter_11849.solverstate')
train(solver, visualize=True)#, ignored=['diff1', 'diff2'])
