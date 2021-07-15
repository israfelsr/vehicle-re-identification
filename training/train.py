# Common imports
import time
import ot
import os
import copy
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax

# Pytorch
import torch
import torch.nn.functional as F
import torch.nn as nn

# Local Imports
import constants as c
from utils import DataLogger, get_corrects

def process_source_one_epoch(feature_extractor_model, classifier_model, optimizer,
                             data_iterator, logger, is_training):
  if is_training:
    feature_extractor_model.train()
    classifier_model.train()
  else:
    feature_extractor_model.eval()
    classifier_model.eval()
  torch.set_grad_enabled(is_training)
  loss_func = nn.CrossEntropyLoss()
  for i, (Xs, ys) in enumerate(data_iterator):
    Xs, ys = Xs.to(device=c.device), ys.to(device=c.device)
    since = time.time()
    if is_training:
      optimizer.zero_grad()  # Reset gradient.
    features_source = feature_extractor_model(Xs)
    ys_pred = classifier_model(features_source)
    loss = loss_func(ys_pred, ys)
    if is_training:
      loss.backward()
      optimizer.step()
    # Running stats.
    batch_size = Xs.shape[0]
    time_batch = time.time() - since
    logger.running_update(acc_source=get_corrects(ys, ys_pred).item(),
                          loss=loss.item(),
                          time=time_batch, examples_count=batch_size)
    if i % c.running_stats_freq == 0 and i > 0:
      logger.print_running(i)
  # Epoch stats.
  logger.epoch_update()
  logger.print_epoch()

def process_target_one_epoch(feature_extractor_model, classifier_model, optimizer,
                             data_iterator, logger, is_training, gamma_criterion, g_criterion):

  def compute_deepjdot_loss(features_source, ys_pred, ys, features_target,
                            yt_pred, gamma_criterion, g_criterion):
    # Compute the euclidian distance in the feature space
    #C0 = cdist(features_source.detach().cpu().numpy(),
    #           features_target.detach().cpu().numpy(), p=0.2)
    C0 = torch.square(torch.cdist(features_source, features_target, p=2.0))
    # Compute the loss function of labels
    #C1 = F.cross_entropy(yt_pred, ys)
    classes = torch.arange(yt_pred.shape[1]).reshape(1, yt_pred.shape[1])
    one_hot_ys = (ys.unsqueeze(1) == classes.to(device=c.device)).float()
    C1 = torch.square(torch.cdist(one_hot_ys, F.softmax(yt_pred, dim=1), p=2.0))
    C = c.alpha * C0 + c.tloss * C1
    # Compute the gamma function
    #gamma = ot.emd(ot.unif(features_source.shape[0]),
    #               ot.unif(features_target.shape[0]), C)
    gamma = ot.emd(
      torch.from_numpy(ot.unif(features_source.shape[0])).to(device=c.device),
      torch.from_numpy(ot.unif(features_target.shape[0])).to(device=c.device),
      C)
    # ot.emd: solve the OT problem for the pdfs of both source and target features
    # ot.unif: return an histogram of the arguments

    # Align Loss
    gamma_loss = gamma_criterion(features_source, features_target, gamma)

    # gamma loss get the fetures of the source, the features of the target
    # and gamma. It first performs the L2 distance between the features and 
    # then return self.jdot_alpha * dnn.K.sum(self.gamma * (gdist))

    # Classifier Loss
    clf_loss = g_criterion(ys, ys_pred, yt_pred, gamma)
    return clf_loss, gamma_loss, clf_loss + gamma_loss
    
  # Set training mode.
  if is_training:
    feature_extractor_model.train()
    classifier_model.train()
  else:
    feature_extractor_model.eval()
    classifier_model.eval()
  torch.set_grad_enabled(is_training)
  for i, (Xs, ys, Xt, yt) in enumerate(data_iterator):
    Xs, ys = Xs.to(device=c.device), ys.to(device=c.device)
    Xt, yt = Xt.to(device=c.device), yt.to(device=c.device)
    since = time.time()
    if is_training:
      optimizer.zero_grad()  # Reset gradient.
    features_source = feature_extractor_model(Xs)
    features_target = feature_extractor_model(Xt)
    ys_pred = classifier_model(features_source)
    yt_pred = classifier_model(features_target)
    
    clf_loss, gamma_loss, loss = compute_deepjdot_loss(features_source,
                                                       ys_pred, ys,
                                                       features_target,
                                                       yt_pred,
                                                       gamma_criterion,
                                                       g_criterion)
    if is_training:
      loss.backward()
      optimizer.step()
    # Running stats.
    batch_size = Xs.shape[0]
    time_batch = time.time() - since
    logger.running_update(acc_source=get_corrects(ys, ys_pred).item(),
                          acc_target=get_corrects(yt, yt_pred).item(),
                          loss=loss.item(), gamma_loss=gamma_loss.item(),
                          clf_loss=clf_loss.item(), time=time_batch,
                          examples_count=batch_size)
    if i % c.running_stats_freq == 0 and i > 0:
      logger.print_running(i)
  # Epoch stats.
  logger.epoch_update()
  logger.print_epoch()

def train_model_(feature_extractor_model, classifier_model, optimizer,
                train_data_iterator, process_one_epoch_func,
                evaluate_data_iterator, train_logger=None, val_logger=None,
                scheduler=None, save_to=None,
                path='content/MyDrive/vehicle-re-identification/training/model_checkpoints',
                num_epochs=25, **kargs):
  # Moving models to correct device
  best_feature_extractor_state_dict = copy.deepcopy(
    feature_extractor_model.state_dict())
  best_classifier_state_dict = copy.deepcopy(classifier_model.state_dict())
  best_acc = 0.0
  best_metrics_str = ""

  feature_extractor_model = feature_extractor_model.to(c.device)
  classifier_model = classifier_model.to(c.device)
  if not train_logger:
    train_logger = DataLogger()
  if not val_logger:
    val_logger = DataLogger(val=True)
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    process_one_epoch_func(feature_extractor_model, classifier_model, optimizer,
                           train_data_iterator, train_logger, is_training=True,
                           **kargs)
    if scheduler:
      scheduler.step()
    if epoch % c.eval_every == 0 and epoch > 0:
      print('Evaluation Pass')
      process_one_epoch_func(feature_extractor_model, classifier_model,
                             optimizer, train_data_iterator, val_logger,
                             is_training=False, **kargs)
      if val_logger.get_last_epoch_total_accuracy() > best_acc:
        best_feature_extractor_state_dict = copy.deepcopy(
          feature_extractor_model.state_dict())
        best_classifier_state_dict = copy.deepcopy(classifier_model.state_dict())
        best_metrics_str = " || ".join(
          ["{}={}".format(key, value_list[-1]) for key, value_list in
           val_logger.epoch_metrics.items()])
        best_acc = val_logger.get_last_epoch_total_accuracy()
        if save_to:
          torch.save({
            'best_feature_extractor_state_dict': best_feature_extractor_state_dict,
            'best_classifier_state_dict': best_classifier_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_logger': train_logger,
            'val_logger': val_logger
            }, os.path.join(path, 'checkpoint_{}.pth'.format(
              save_to)))
  print("Best validation results:")
  print(best_metrics_str)
  return (best_feature_extractor_state_dict, best_classifier_state_dict,
          train_logger, val_logger)

# TODO: remove?
def train_one_epoch(feature_extractor_model, classifier_model, optimizer,
                    gamma_criterion, g_criterion, source_loader, target_iter,
                    logger):
  feature_extractor_model.train()
  classifier_model.train()

  for t, (Xs, ys) in enumerate(source_loader['train']):
    since = time.time()
    # Getting batch of source and target data
    Xt, yt = target_iter['train'].get_next()
    Xs, ys = Xs.to(device=c.device), ys.to(device=c.device)
    Xt, yt = Xt.to(device=c.device), yt.to(device=c.device)
    
    optimizer.zero_grad()
    # Extracting Features
    feat_source = feature_extractor_model(Xs)
    feat_target = feature_extractor_model(Xt)

    # Getting Predictions
    ys_pred = classifier_model(feat_source)
    yt_pred = classifier_model(feat_target)
    
    # Compute the euclidian distance in the feature space
    C0 = cdist(feat_source.detach().cpu().numpy(), feat_target.detach().cpu().numpy(), p=2.0)
    # Compute the loss function of labels
    C1 = F.cross_entropy(yt_pred, ys)
    C = c.alpha * C0 + c.beta * C1.item()

    # Compute the gamma function
    gamma = ot.emd(ot.unif(feat_source.shape[0]), ot.unif(feat_target.shape[0]), C)
    # ot.emd: solve the OT problem for the pdfs of both source and target features
    # ot.unif: return an histogram of the arguments

    # Align Loss
    gamma_loss = gamma_criterion(feat_source, feat_target, torch.from_numpy(gamma).to(device=c.device))

    # gamma loss get the fetures of the source, the features of the target
    # and gamma. It first performs the L2 distance between the features and 
    # then return self.jdot_alpha * dnn.K.sum(self.gamma * (gdist))

    # Classifier Loss
    clf_loss = g_criterion(ys, ys_pred, yt_pred, torch.from_numpy(gamma).to(device=c.device))
    loss = clf_loss + gamma_loss
    
    # Barckward pass
    loss.backward()
    optimizer.step()
    
    # Running Stats
    bsize = Xs.shape[0]
    acc_s, acc_t = get_corrects(ys, ys_pred, yt, yt_pred)
    time_batch = time.time() - since
    logger.running_update(acc_s.item(), acc_t.item(), gamma_loss.item(), 
                                clf_loss.item(), loss.item(), time_batch, bsize)
    if t % c.running_stats_freq == 0:
      logger.print_running(t)
  # Epoch Stats
  logger.epoch_update()
  logger.print_epoch()

def evaluate(g_model, f_model, gamma_criterion, g_criterion, source_loader,
             target_iter, logger):
  g_model.eval()
  f_model.eval()
  with torch.no_grad():
    for t, (Xs, ys) in enumerate(source_loader['val']):
      since = time.time()
      # Getting batch of source and target data
      Xt, yt = target_iter['val'].get_next()
      Xs, ys = Xs.to(device=c.device), ys.to(device=c.device)
      Xt, yt = Xt.to(device=c.device), yt.to(device=c.device)

      # Extracting Features
      feat_source = g_model(Xs)
      feat_target = g_model(Xt)

      # Getting Predictions
      ys_pred = f_model(feat_source)
      yt_pred = f_model(feat_target)

      # Computing loss
      C0 = cdist(feat_source.detach().cpu().numpy(), feat_target.detach().cpu().numpy(), p=2.0)
      # Compute the loss function of labels
      C1 = F.cross_entropy(yt_pred, ys)
      C = c.alpha * C0 + 1.0 * C1.item()
      gamma = ot.emd(ot.unif(feat_source.shape[0]), ot.unif(feat_target.shape[0]), C)
      gamma_loss = gamma_criterion(feat_source, feat_target, torch.from_numpy(gamma).to(device=c.device))
      clf_loss = g_criterion(ys, ys_pred, yt_pred, torch.from_numpy(gamma).to(device=c.device))
      loss = gamma_loss + clf_loss

      # Running Stats
      bsize = Xs.shape[0]
      acc_s, acc_t = get_corrects(ys, ys_pred, yt, yt_pred)
      time_batch = time.time() - since
      logger.running_update(acc_s.item(), acc_t.item(), gamma_loss.item(), 
                            clf_loss.item(), loss.item(), time_batch, bsize)
    # Epoch Stats
    logger.epoch_update()
    logger.print_epoch()

# TODO: remove?
def train_model(g_model, f_model, optimizer, gamma_criterion, g_criterion,
                source_loader, target_iter, train_logger=None, scheduler=None,
                save=True, path=None, num_epochs=25):
  # Moving models to correct device
  if save:
    best_g_model = copy.deepcopy(g_model.state_dict())
    best_f_model = copy.deepcopy(f_model.state_dict())
    best_acc = 0.0

  g_model = g_model.to(c.device)
  f_model = f_model.to(c.device)
  if not train_logger:
    train_logger = DataLogger()
  val_logger = DataLogger(val=True)
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    train_one_epoch(g_model, f_model, optimizer, gamma_criterion, g_criterion,
                    source_loader, target_iter, train_logger)
    if scheduler:
      scheduler.step()
    if epoch % c.eval_every == 0 and epoch > 0:
      print('Evaluation Pass')
      evaluate(g_model, f_model, gamma_criterion, g_criterion, source_loader,
               target_iter, val_logger)
      if (val_logger.epoch_metrics['acc_source'][-1] +
          val_logger.epoch_metrics['acc_target'][-1]) > best_acc:
        best_g_model = copy.deepcopy(g_model.state_dict())
        best_f_model = copy.deepcopy(f_model.state_dict())
        if save:
          torch.save({
            'g_model_state_dict': best_g_model,
            'f_model_state_dict': best_f_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_logger': train_logger,
            'val_logger': val_logger
            }, os.path.join(path, 'training/weights/checkpoint.pth'))
  if save:
    return best_g_model, best_f_model, train_logger, val_logger
  else:
    return train_logger, val_logger