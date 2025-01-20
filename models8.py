import torch.nn as nn
import torch.nn.functional as F
import torch
from dynamic_conv import Dynamic_conv2d
import numpy as np



class ResidualBlock(nn.Module):

    def __init__(self, in_features, norm=False):
        super(ResidualBlock, self).__init__()

        block = [nn.ReflectionPad2d(1),
                 nn.Conv2d(in_features, in_features, 3),
                 # nn.InstanceNorm2d(in_features),
                 nn.ReLU(inplace=True),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(in_features, in_features, 3),
                 # nn.InstanceNorm2d(in_features)
                 ]

        if norm:
            block.insert(2, nn.InstanceNorm2d(in_features))
            block.insert(6, nn.InstanceNorm2d(in_features))

        self.model = nn.Sequential(*block)

    def forward(self, x):
        return x + self.model(x)



class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y




class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate): #nchannels = 256 denselayer = 6 growthrate = 32
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out




class GeneratorA2B(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorA2B, self).__init__()

        # First Layer
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        in_features = 256
        #Residual (bottleneck)

        self.model = nn.Sequential(RDB(in_features, n_residual_blocks, 32),
                                   CALayer(in_features),
                                   PALayer(in_features))
        self.model2 = nn.Sequential(RDB(in_features, n_residual_blocks, 32),
                                   CALayer(in_features),
                                   PALayer(in_features))

        # Upsample layers
        self.conv_5 =nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        # Final Layer
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        )

        self.ka = nn.Sequential(
            nn.Conv2d(input_nc, 64, 7, padding=3, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 8, 1, padding=0, bias=True),
            #nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, output_nc, 1, padding=0, bias=True),
            nn.Sigmoid()
        )



    def forward(self, x):
        ka = self.ka(x)
        h1 = self.conv_1(x)
        h2 = self.conv_2(h1)
        h3 = self.conv_3(h2)

        res1 = self.model(h3)
        res2 = self.model2(res1)
        avg1 = self.avg1(h3)
        h45 = res2 * avg1

        h4 = self.conv_5(h45)

        h5 = self.conv_6(h4)

        h6 = self.conv_7(h5)
        img = torch.mul((1 - h6), ka) + torch.mul(h6, x)
        return img

    def create_gaussian_kernel(self, kernel_size, sigma):
        kern1d = torch.Tensor([np.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        kern1d = kern1d / kern1d.sum()
        kern2d = kern1d.reshape(-1, 1).mm(kern1d.reshape(1, -1))
        #kern2d = kern2d.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        #kern2d = kern2d.repeat(1, in_channels, 1, 1).cuda()
        #print(kern2d)
        return kern2d



class GeneratorB2A(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorB2A, self).__init__()

        # First Layer
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True),
                 nn.Dropout2d(0.2)]

        # Downsapling Layers
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2),
                      #nn.MaxPool2d(2) #這個要刪掉，不然輸出圖像會變小
                      ]
            in_features = out_features
            out_features = in_features*2

        #Residual (bottleneck)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)
                      ]

        # Upsample layers
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True),
                      nn.Dropout2d(0.2)]
            in_features = out_features
            out_features = in_features//2

        # Final Layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # 4*Conv layers coupled with leaky-relu & instance norm.
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # Final layer.
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
