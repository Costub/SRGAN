import torch
from torch import nn
from torchsummary import summary




class res_block(nn.Module):

  def __init__(self):
    super(res_block,self).__init__()

    self.block = nn.Sequential(

        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(64),
        nn.PReLU(),

        nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(64),
        nn.PReLU()
    )
  
  def forward(self,x):
    return ( self.block(x) + x )




class generator(nn.Module):

  def __init__(self,b=16):
    super(generator,self).__init__()

    self.gen1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=9,padding=4,stride=1),
        nn.PReLU()
    )
        
    resblock = [res_block() for _ in range(b)]
    self.res = nn.Sequential(*resblock)

    self.gen2=  nn.Sequential(
        
        nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
        nn.PixelShuffle(2),
        nn.PReLU(),

        nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
        nn.PixelShuffle(2),
        nn.PReLU(),

        nn.Conv2d(64, 3, kernel_size=9, padding=4,stride=1)
    )

  def forward(self,x):
    x = self.gen1(x)
    x = self.res(x)
    x = self.gen2(x)
    return x




class discriminator(nn.Module):

  def __init__(self, im_shape=256):
    super(discriminator, self).__init__()

    x = int(im_shape*im_shape*2)

    self.disc = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            nn.Linear(x,1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 1),
            nn.Sigmoid()

        )

  def forward(self, x):
        return self.disc(x)