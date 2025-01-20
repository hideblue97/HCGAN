#!/usr/bin/python3

import argparse
import itertools
import os
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn

from models8 import GeneratorA2B, GeneratorB2A
from models8 import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from utils import VGGNet

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=60,
                    help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8,
                    help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/Haze2Dehaze/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=40,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3,
                    help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4,
                    help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Networks
netG_A2B = GeneratorA2B(opt.input_nc, opt.output_nc)
netG_B2A = GeneratorB2A(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(torch.cuda.device_count())
    netG_A2B = nn.DataParallel(netG_A2B, device_ids=device_ids).to(device)
    netG_B2A = nn.DataParallel(netG_B2A, device_ids=device_ids).to(device)
    netD_A = nn.DataParallel(netD_A, device_ids=device_ids).to(device)
    netD_B = nn.DataParallel(netD_B, device_ids=device_ids).to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# pretrained VGG19 module set in evaluation mode for feature extraction
vgg = VGGNet().cuda().eval()


def perceptual_loss(x, y):
    c = nn.MSELoss()

    rx = netG_B2A(netG_A2B(x))
    ry = netG_A2B(netG_B2A(y))

    fx1, fx2 = vgg(x)
    fy1, fy2 = vgg(y)

    frx1, frx2 = vgg(rx)
    fry1, fry2 = vgg(ry)

    m1 = c(fx1, frx1)
    m2 = c(fx2, frx2)

    m3 = c(fy1, fry1)
    m4 = c(fy2, fry2)



    loss = (m1 + m2 + m3 + m4) * 5 * 0.5

    return loss


# Lossess
criterion_GAN = torch.nn.MSELoss()  # Adversarial Loss
criterion_cycle = torch.nn.L1Loss()  # Cyclic consistency loss


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0),
                       requires_grad=False)  # real
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0),
                       requires_grad=False)  # fake

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader with data augmentations
transforms_ = [transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)


if os.path.exists('output/netG_A2B.pth'):
    checkpoint = torch.load('output/netG_A2B.pth')
    netG_A2B.load_state_dict(checkpoint['model'])
    optimizer_G.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netG_A2B epoch {} success£¡'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netG_A2B model, will start training from scratch')

if os.path.exists('output/netG_B2A.pth'):
    checkpoint = torch.load('output/netG_B2A.pth')
    netG_B2A.load_state_dict(checkpoint['model'])
    optimizer_G.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netG_B2A epoch {} success£¡£¡'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netG_B2A model, will start training from scratch')

if os.path.exists('output/netD_A.pth'):
    checkpoint = torch.load('output/netD_A.pth')
    netD_A.load_state_dict(checkpoint['model'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netD_A epoch {} success£¡'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netD_A model, will start training from scratch')

if os.path.exists('output/netD_B.pth'):
    checkpoint = torch.load('output/netD_B.pth')
    netD_B.load_state_dict(checkpoint['model'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Load netD_B epoch {} successe!'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved netD_B model, will start training from scratch')


# Loss plot
logger = Logger(start_epoch, opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(start_epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        #print(real_A.size())
        #print(real_B.size())

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A)
        #print(fake_B.size())
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*5.0

        fake_A = netG_B2A(real_B)
        #print(fake_A.size())
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*5.0

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        #print(recovered_A.size())
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)#*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)#*10.0

        # Perceptual loss
        loss_perceptual = perceptual_loss(real_A, real_B)

        # Total loss
        loss_G = loss_perceptual + loss_GAN_A2B + \
            loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        # logged using visdom
        logger.log({'loss_G': loss_G, 'loss_G_perceptual': loss_perceptual, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    save_folder = Path("./output")
    # Save models checkpoints
    '''torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')'''
    state = {'model': netG_A2B.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / "netG_A2B.pth")
    state = {'model': netG_B2A.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netG_B2A.pth')
    state = {'model': netD_A.state_dict(), 'optimizer': optimizer_D_A.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netD_A.pth')
    state = {'model': netD_B.state_dict(), 'optimizer': optimizer_D_B.state_dict(), 'epoch': epoch}
    torch.save(state, save_folder / 'netD_B.pth')
###################################
# python train.py --dataroot datasets/smoke/ --cuda
# python test.py --dataroot datasets/smoke/ --cuda
# python -m visdom.server
