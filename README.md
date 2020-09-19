# Dilated VGG Networks (Custom Dataset for Image Classification task)

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like Dilated VGG-16, Dilated VGG-19, VGG-16 and VGG-19 were used for the task of Dog-Cat image classification. The input to the CNN networks was a (224 x 224 x 3) image and the number of classes were 2, where '0' was for a cat and '1' was for a dog. The CNN architectures were implemented in PyTorch and the loss function was Cross Entropy Loss. The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), momentum(m), weight decay(wd) and batch size(bs).


### Data
The data for the task of Dog-Cat image classification can be downloaded from: https://drive.google.com/drive/folders/1EdVqRCT1NSYT6Ge-SvAIu7R5i9Og2tiO?usp=sharing. The dataset has been divided into three sets: Training data, Validation data and Testing data. The analysis of different CNN architectures for Dog-Cat image classification was done on comparing the Training Accuracy and Validation Accuracy values.


### Results
The results after using different CNN architectures are given below:

1. <b>VGG-16(pretrained on ImageNet dataset)</b><br>

Training Accuracy = 99.27% and Validation Accuracy = 96.73% (e = 50, lr = 0.005, m = 0.9, bs = 32, wd = 0.001)<br>


2. <b>VGG-19(pretrained on ImageNet dataset)</b><br>

Training Accuracy = 99.13% and Validation Accuracy = 97.25% (e = 50, lr = 0.005, m = 0.9, bs = 32, wd = 5e-4)<br>


3. <b>Dilated VGG-16</b><br>

Training Accuracy = 99.17% and Validation Accuracy = 97.11% (e = 40, lr = 1e-3, m = 0.9, bs = 32, wd = 5e-4)<br>


4. <b>Dilated VGG-19</b><br>

Training Accuracy = 98.81% and Validation Accuracy = 97.18% (e = 40, lr = 1e-3, m = 0.9, bs = 32, wd = 5e-4)<br>


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://www.tandfonline.com/doi/full/10.1080/24699322.2019.1649071
2. https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5
