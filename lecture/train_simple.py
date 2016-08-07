import caffe
import matplotlib.pyplot

# Initialize solver
caffe.set_mode_gpu()
solver = caffe.SGDSolver('practice1_solver.pt')

class LossManager(object):
  def __init__(self, solver):
    # Initialize figure
    #self.fig, self.axes = matplotlib.pyplot.subplots()
    #self.fig.show()
    self.loss_list = []
    self.window_size = 10
    self.window_start = 0
    self.initial_lr = solver.get_base_lr()

  def add_current_loss(self, loss):
    if len(self.loss_list) == 0:
      mean_loss = loss
    else:
      mean_loss = 0.01 * loss + 0.99 * self.loss_list[-1]
    self.loss_list.append(mean_loss)

  def plot_loss(self):
    self.axes.clear()
    self.axes.plot(range(self.loss_list), self.loss_list)
    self.fig.canvas.draw()
    matplotlib.pyplot.pause(0.01)

  def update_base_lr(self, solver):
    if len(self.loss_list) - self.window_start > self.window_size \
       and self.loss_list[-1] > 0.999 * self.loss_list[-self.window_size]:
      print 'base_lr reduced at iter %d: %.3f > %.3f' % (solver.iter, self.loss_list[-1], 0.999 * self.loss_list[-self.window_size])
      solver.set_base_lr(solver.get_base_lr() * 0.5)
      self.window_size *= 2
      self.window_start = solver.iter
    if solver.get_base_lr() < 1e-3:
      print 'base_lr %.3f < 1e-3. Reset base_lr and start next round at iter %d' % (solver.get_base_lr(), solver.iter)
      solver.set_base_lr(self.initial_lr)
      self.window_size = 10
      self.window_start = solver.iter

manager = LossManager(solver)

while True:
  solver.step(1)

  # Add loss
  loss = solver.net.blobs['loss'].data.flatten()
  manager.add_current_loss(loss)
  manager.update_base_lr(solver)

  # Update plot every 5 iterations
  if False and solver.iter % 5 == 0:
    manager.plot_loss()
