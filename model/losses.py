import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as c

class OtLoss(nn.Module):
  '''
  Performs the loss to align the distributions
  '''
  def __init__(self, alpha, align_weight):
    super().__init__()
    self.alpha = alpha
    self.align_weight = align_weight

  def forward(self, ft_source, ft_target, gamma):
    num_features = ft_target.shape[1]
    l2_dist = (torch.sum(torch.square(ft_source),1).view((-1,1)) +
               torch.sum(torch.square(ft_target),1).view((1,-1)))
    l2_dist -= 2.0*torch.mm(ft_source, ft_target.view(num_features, c.batch_size))
    return self.align_weight * self.alpha * torch.sum(gamma * l2_dist)

class ClfLoss(nn.Module):
  def __init__(self, classifier_weight, tloss, sloss, device='cpu'):
    super().__init__()
    self.classifier_weight = classifier_weight
    self.tloss = tloss
    self.sloss = sloss
    self.eps = 1e-9
    self.cross_entorpy = nn.CrossEntropyLoss()
    self.device = device

  def forward(self, ys, ys_pred, yt_pred, gamma):
    cross_entropy_source = self.cross_entorpy(ys_pred, ys)
    classes = torch.arange(c.num_class).reshape(1, c.num_class)
    one_hot_ys = (ys.unsqueeze(1) == classes.to(device=self.device)).float()
    yt_pred_ = yt_pred
    yt_pred = -nn.LogSoftmax(dim=1)(yt_pred)
    loss = torch.matmul(one_hot_ys, yt_pred.t())
    return self.classifier_weight * (self.tloss * torch.sum(gamma * loss) +
                                     self.sloss * cross_entropy_source)