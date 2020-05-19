from resnet import *
import torch
from torch.autograd import Variable
import torch.optim as optim

model = ResNet50(num_classes=4).cuda()
