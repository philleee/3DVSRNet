import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data as data
from PIL import Image
import os.path
import random

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0
    Returns
    -------
        list
            2D list described above.
    """

    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
class mydataset(Dataset):
  def __init__(self,path, transform=None):
    self.targetpath = path
    self.transform = transform
  
  def __getitem__(self, idx):
    data = []
    img0 = _pil_loader(path=self.targetpath+'/'+str("%05d" % (idx))+'.jpg')
    imgT = _pil_loader(path=self.targetpath+'/'+str("%05d" % (idx+1))+'.jpg')
    img1 = _pil_loader(path=self.targetpath+'/'+str("%05d" % (idx+2))+'.jpg')
    if self.transform:
      img0 = self.transform(img0)
      imgT = self.transform(imgT)
      img1 = self.transform(img1)
    return [img0,imgT,img1],1
  def __len__(self):
    count=0
    for _ in os.listdir(self.targetpath):
      count+=1
    return count-2
    
if __name__ == "__main__":
  mydataset = mydataset('../../data/imgs/jpg_train')
  print(len(mydataset))
  imgs,one = mydataset[0]
  print(imgs[0].size)
  print(imgs[1].size)
  print(imgs[2].size)