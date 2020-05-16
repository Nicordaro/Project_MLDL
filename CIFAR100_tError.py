from PIL import Image
import os
import os.path
import numpy as np
import pickle



from .vision import VisionDataset
from .utils import check_integrity, download_and_extract_archive
from torchvision.datasets import CIFAR100

def pil_loader(f):
    # open file
    img = Image.open(f)
    return img.convert('RGB')


class CIFAR100_tError():
  def __init__(self, root, train=True, transform=None, traget_transform=None, download=False, lbls=[0:100]):
    self.prova = CIFAR100(self, root, train=True, transform=None, traget_transform=None, download=False)
    
    self.data = []
    self.labels = []
      for element in self.prova:
        image, label = element
        if label in lbls:
          self.data.append(image)
          self.labels.append(label)
          
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        
        image = pil_loader(self.data[index])
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)
        return length
      
    def increment(self, newlbls):
      for element in self.prova:
        image, label = element
        if label in newlbls:
          self.data.append(image)
          self.labels.append(label)
      
