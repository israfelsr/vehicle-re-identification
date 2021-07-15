# Common imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
# Local Files
import constants as c

# Veri dataset loader
def get_common(df1, df2):
  '''
  Return df1 and df2 with only common labels
  '''
  common_labels = list(set(df1.label.unique()) & set(df2.label.unique()))
  df1 = df1.loc[df1['label'].isin(common_labels)]
  df2 = df2.loc[df2['label'].isin(common_labels)]
  return df1, df2

def split_dataframe(data, train):
  msk = np.random.rand(len(data)) < 0.8
  train = data[msk]
  test = data[~msk]
  return train, test

def load_dataframe(root_dir, source_cam, target_cam, split=True):
  '''
  Using two different sets of cameras create source and target dataframe
  '''
  df = pd.read_csv(os.path.join(root_dir, 'name_train.txt'), delimiter='\n',
                   header=None, names=['files'])
  df[['label', 'cam']] = df['files'].str.split('_', n=2, expand=True).iloc[:,0:2]
  df.label = df.label.astype(int)
  source_df = df.loc[df['cam'].isin(source_cam)]
  target_df = df.loc[df['cam'].isin(target_cam)]
  del df
  if split:
    s_train, s_test = split_dataframe(source_df, 0.8)
    t_train, t_test = split_dataframe(target_df, 0.8)
    s_train, s_val = split_dataframe(s_train, 0.8)
    t_train, t_val = split_dataframe(t_train, 0.8)
    s_train, t_train = get_common(s_train, t_train)
    s_test, t_test = get_common(s_test, t_test)
    s_val, t_val = get_common(s_val, t_val)
    return s_train, s_test, s_val, t_train, t_test, t_val
  else:
    source_df, target_df = get_common(source_df, target_df)
    return  source_df, target_df

class VeRi(Dataset):
  '''
  Dataset generator with outputs:
      - x_si = image from a labeled dataset
      - y_si = label of the image x_i
      - x_tj = image of the target dataset
  '''
  def __init__(self, df, root_dir, src='source', mode='train', transform=None):
    self.mode = mode
    self.transform = transform
    self.df = df
    self.src = src
    self.files = self.df.files
    self.img_dir = os.path.join(root_dir, 'image_train')
        
  def __len__(self):
    return len(self.files)   

  def __getitem__(self, idx):
    X = Image.open(os.path.join(self.img_dir, 
                                self.df.files.iloc[idx]))
    if self.transform:
      X = self.transform(X)
    if self.src == 'source':
      label = self.df.label.iloc[idx]
      y = torch.tensor(label)
      return X, y
    else:
      return X

def load_dataset(root, source_cam, target_cam, split=True, target_label=True):
  '''
  Function to create the dataset.
  Inputs:
    - split: Splits the dataset into train, test and validation
    - target_label: return the target dataset with the labels 
  '''
  tfrm = transforms.Compose([transforms.Resize(size=(128,128)), 
                             transforms.ToTensor()])
  if split:
    s_train, s_test, s_val, t_train, t_test, t_val = load_dataframe(root, 
                                                                    source_cam=source_cam,
                                                                    target_cam=target_cam,
                                                                    split=split)
    source_train_gen = VeRi(s_train, root, transform=tfrm)
    source_test_gen = VeRi(s_test, root, mode='test', transform=tfrm)
    source_val_gen = VeRi(s_val, root, transform=tfrm)
    
    if target_label:
      target_train_gen = VeRi(t_train, root, transform=tfrm)
      target_test_gen = VeRi(t_test, root, mode='test', transform=tfrm)
      target_val_gen = VeRi(t_val, root, transform=tfrm)
    else:
      target_train_gen = VeRi(t_train, root, src='target', transform=tfrm)
      target_test_gen = VeRi(t_test, root, src='target', mode='test',
                             transform=tfrm)
    
    return source_train_gen, source_test_gen, source_val_gen, \
             target_train_gen, target_test_gen, target_val_gen
  else:
    data_gen = JointDataset(s_df, t_df, root, transform=tfrm)
    return data_gen

class QueryDataset(Dataset):
  def __init__(self, df_file, img_dir, transform=None):
    self.df = pd.read_csv(os.path.join(root, df_file), delimiter='\n', 
                          header=None, names=['files'])
    self.img_dir = img_dir
    self.transform = transform
    self.files = self.df.files

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    X = Image.open(os.path.join(self.img_dir, 
                                self.df.files.iloc[idx]))
    if self.transform:
      X = self.transform(X)
    return X

# TODO: remove?
class DataloaderIterator():
  def __init__(self, dataloader):
    self.iterator = iter(dataloader)
    self.dataloader = dataloader

  def get_next(self):
    try:
      X, y = next(self.iterator)
    except StopIteration:
      self.iterator = iter(self.dataloader)
      X, y = next(self.iterator)
    return X, y

class SourceTargetIterator():
  def __init__(self, source_dataloader, target_dataloader):
    self.source_dataloader = source_dataloader
    self.target_dataloader = target_dataloader

  def __iter__(self):
    self.source_iterator = iter(self.source_dataloader)
    self.target_iterator = iter(self.target_dataloader)
    return self
  
  def _get_next_target(self):
    try:
      return next(self.target_iterator)
    except StopIteration:
      self.target_iterator = iter(self.target_dataloader)
      return next(self.target_iterator)

  def __next__(self):
    # When source is finished it will raise the StopIteration exception.
    Xs, ys = next(self.source_iterator)
    Xt, yt = self._get_next_target()
    return Xs, ys, Xt, yt


def source_and_target_epoch_iterator(source_dataloader, target_dataloader):
  target_iterator = iter(target_dataloader)
  for Xs, ys in source_dataloader:
    try:
      Xt, yt = next(target_iterator)
    except StopIteration:
      target_iterator = iter(target_dataloader)
      Xt, yt = next(target_iterator)
    Xs, ys = Xs.to(device=c.device), ys.to(device=c.device)
    Xt, yt = Xt.to(device=c.device), yt.to(device=c.device)
    yield Xs, ys, Xt, yt