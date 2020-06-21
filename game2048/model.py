import torch 
import torch.nn as nn

class CNN(nn.Module):
     def __init__(self):
          super(CNN, self).__init__()
          self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(4,1),stride=1, padding=0),
                    nn.ReLU(),
                    )
          self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(1,4), stride=1, padding=0),
                    nn.ReLU(),
                    )
          self.layer3 = nn.Sequential(
            	     nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(2,2), stride=1, padding=0),
                     nn.ReLU(), 
          )
          self.layer4 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(3,3), stride=1, padding=0), 
                     nn.ReLU(),
          )
          self.layer5 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(4,4), stride=1, padding=0),
                     nn.ReLU(),
          )
          self.layer6 = nn.Sequential(
               nn.Linear(2816,512),
               nn.ReLU()
          )
          self.layer7 = nn.Sequential(
               nn.Linear(512,128), 
               nn.ReLU()
          )
          self.layer8 = nn.Sequential(
               nn.Linear(128,4),
               nn.ReLU()
          )

     def forward(self, x):
         x1 = self.layer1(x)
         x1 = x1.view(x1.size(0), -1)
         x2 = self.layer2(x)
         x2 = x2.view(x2.size(0), -1)
         x3 = self.layer3(x)
         x3 = x3.view(x3.size(0), -1)
         x4 = self.layer4(x)
         x4 = x4.view(x4.size(0), -1)
         x5 = self.layer5(x)
         x5 = x5.view(x5.size(0), -1)
         xcat = torch.cat((x1, x2, x3, x4, x5), 1)
         x = self.layer6(xcat)
         x = self.layer7(x)
         x = self.layer8(x)
         return x


