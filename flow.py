
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import mydataset
from math import log10
import datetime
from tensorboardX import SummaryWriter
import time

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--checkpoint_dir", type=str, default='checkpoint', help='path to folder for saving checkpoints')
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=30, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=16, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=16, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[10,20], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=10, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
args = parser.parse_args()



###Initialize flow computation and arbitrary-time flow interpolation CNNs.

if torch.cuda.is_available():
  print('cuda is Running!')
else:
  print('cuda is not Running!')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = model.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = model.UNet(20, 5)
ArbTimeFlowIntrp.to(device)

dict1 = torch.load(args.checkpoint_dir + "/3DVSRNet3.ckpt")
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])
###Initialze backward warpers for train and validation datasets


trainFlowBackWarp      = model.backWarp(224, 224, device)
trainFlowBackWarp      = trainFlowBackWarp.to(device)
validationFlowBackWarp = model.backWarp(224, 224, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)


###Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])
validationloader = torch.utils.data.DataLoader(
    mydataset.mydataset('old_data', transform=transform),
    # mydataset.mydataset('../../../data/imgs/linear/jpg', transform=transform),
    batch_size=1,
    shuffle=True,
)

###Create transform to display image from tensor


negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False



def validate():
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)
                        
            
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]
            a = torch.cat((F_0_1,I0-I0),1)[:,:3,:,:]
            print(a.shape)
            print(I1.shape)

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            
            break
            
    return revNormalize(frame0.cpu()[0]),revNormalize(frameT.cpu()[0]),revNormalize(frame1.cpu()[0]),revNormalize(a.cpu()[0])
count = 0
while(count<5):
  img0,imgT,img1,a = validate()
  # a = torch.squeeze(a.cpu().clone())
  print(a.shape)
  from PIL import Image
  import matplotlib.pyplot as plt
   
  loader = transforms.Compose([
      transforms.ToTensor()])  
   
  unloader = transforms.ToPILImage()
  
  fig = plt.figure()
  ax = fig.add_subplot(231)
  
  image = img0.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  ax.imshow(image)
  
  ax = fig.add_subplot(232)
  image4 = imgT.cpu().clone()
  image4 = image4.squeeze(0)
  image4 = unloader(image4)
  ax.imshow(image4)
  
  ax = fig.add_subplot(233)
  image1 = img1.cpu().clone()
  image1 = image1.squeeze(0)
  image1 = unloader(image1)
  ax.imshow(image1)
  
  ax = fig.add_subplot(235)
  image2 = a.cpu().clone()
  image2 = image2.squeeze(0)
  image2 = unloader(image2)
  ax.imshow(image2)
  
  
  
  plt.show()
  # plt.savefig('new_img_data/train/img'+str(count)+'.jpg')
  count+=1

