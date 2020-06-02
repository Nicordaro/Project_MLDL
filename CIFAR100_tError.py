import os
import os.path
import numpy as np
import pickle
import torch

from sklearn import preprocessing
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets import CIFAR100
# from .utils import check_integrity, download_and_extract_archive


def pil_loader(f):
    # open file
    #img = Image.open(f)
    return f.convert('RGB')


class CIFAR100_tError(CIFAR100):
  """
  This class implements some custom useful functions along with the original
  CIFAR100 class functions
  """

  def __init__(self, root, train=True, transform=None,download=False, lbls=[]):
    """
    When a dataset is created by mean of CIFAR100_tError all the images are in
    self.dataset
    The actual dataset is self.data along with self.labels that start empty and
    are populated by the function increment
    """
    flag = train
    self.dataset = CIFAR100(root, train=flag, download=download)
    self.transform = transform
    self.data = []
    self.labels = []
    self.encoded = []


  def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)
        return length
      
          
  def __getitem__(self, index):
      '''
      __getitem__ should access an element through its index
      Args:
          index (int): Index
      Returns:
          tuple: (sample, target) where target is class_index of the target class.
      '''
      
      image = pil_loader(self.data[index])
      label = self.encoded[index]
      
      if self.transform is not None:
          image = self.transform(image)
      
      return image, label
    

  def increment(self, newlbls, mapping):
    """
    This function accepts as arguments a list of labels (newlbls) whose corresponding images
    are to be added to the dataset (self.data), and a mapping list (mapping) to remap
    the labels 
    """
    for element in self.dataset:
      image, label = element
      if label in newlbls:
        self.data.append(image)
        self.labels.append(label)
        self.encoded.append(mapping[newlbls.index(label)])
    return
