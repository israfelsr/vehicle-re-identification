import matplotlib.pyplot as plt

# Pytorch
import torch

# Local Imports
import constants as c

class DataLogger():
  def __init__(self, val=False):
    self.val = val
    self.epoch_metrics = {
        'acc_source' : [],
        'acc_target' : [],
        'gamma_loss' : [],
        'clf_loss' : [],
        'loss' : []
    }
    self.running_metrics = {
        'time' : 0,
        'examples_count' : 0,
        'acc_source' : 0,
        'acc_target' : 0,
        'gamma_loss' : 0,
        'clf_loss' : 0,
        'loss' : 0,
    }

  def epoch_update(self):
    for key in self.epoch_metrics:
      self.epoch_metrics[key].append(
        self.running_metrics[key] / self.running_metrics['examples_count'])
    self.running_metrics = {
        'time' : self.running_metrics['time'],
        'examples_count' : 0,
        'acc_source' : 0,
        'acc_target' : 0,
        'gamma_loss' : 0,
        'clf_loss' : 0,
        'loss' : 0,
    }
  
  def running_update(self, **kargs):
    for key, value in kargs.items():
      self.running_metrics[key] += value
  
  def print_running(self, t):
    print('Iter {}, Running Stats:'.format(t))
    print(" || ".join(["{}={}".format(key, value) for
                       key, value in self.running_metrics.items()]))
  
  def print_epoch(self):
    if self.val:
      print('Validation Results')
    else:
      print('Epoch Results:')
    print(" || ".join(["{}={}".format(key, value_list[-1]) for
                       key, value_list in self.epoch_metrics.items()]))
    print('-' * 10)

  def get_last_epoch_total_accuracy(self):
    acc_source = 0
    if self.epoch_metrics['acc_source']:
      acc_source = self.epoch_metrics['acc_source'][-1] 
    acc_target = 0
    if self.epoch_metrics['acc_target']:
      acc_target = self.epoch_metrics['acc_target'][-1]
    return acc_source + acc_target
  
  def plot_accuracy(self):
    plt.figure()
    plt.plot(self.epoch_metrics['acc_target'])
    plt.plot(self.epoch_metrics['acc_source'])
    plt.legend(['acc_target','acc_source'])
    if self.val:
      plt.title('Validation: Accuracies vs Epochs')
    else:
      plt.title('Training: Accuracies vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Value')
    plt.show()

  def plot_loss(self):
    plt.figure()
    plt.plot(self.epoch_metrics['loss'])
    plt.plot(self.epoch_metrics['gamma_loss'])
    plt.plot(self.epoch_metrics['clf_loss'])
    plt.legend(['loss', 'gamma_loss', 'clf_loss'])
    if self.val:
      plt.title('Validation: Loss vs Epochs')
    else:
      plt.title('Training: Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.show()
      
def get_corrects(y, y_pred):
  _, y_pred = torch.max(y_pred, 1)
  corrects = torch.sum(y_pred == y.data)
  return corrects