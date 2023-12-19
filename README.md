This is the code used for study 'A Spatio-Temporal Domain Refinement Method for Geological Models Based on Video Super-Resolution'

model.py is used to create 3DVSRNet, which segments the input image into different categories and defines the structure of the encoder and decoder, including convolutional layers, pooling layers, and deconvolutional layers, among others. It also defines a "down" class and an "up" class. The "down" class includes average pooling layers, convolutional layers, and Leaky ReLU layers, which are used to reduce the resolution of the feature maps. The "up" class includes bilinear interpolation layers, convolutional layers, and Leaky ReLU layers, which are used to restore the resolution of the feature maps to the original image size. Additionally, the "backWarp" class is used to implement image reverse mapping.        

train.py is used to train 3DVSRNet. It includes loading the dataset, defining the model, defining the loss function, and the process of training and validation.

mydataset.py provides a custom dataset class that can be used for loading and preprocessing the geological profiles data, and it offers several data manipulation methods.
