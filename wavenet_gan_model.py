from __future__ import division
import os, sys, time
import math
import scipy
from scipy import integrate, constants
import torch
from torch import nn, optim
from torch.autograd import grad
import autograd.numpy as np
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger

#First attempt at a single discriminator network to learn a complex function

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, coords):
        super(DiscriminatorNet, self, coords).__init__()
        n_features = 2
        n_out = 2
        x, t = np.split(coords, 2)
#Hope that the above splits the input into x, t coordinates

        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(100, n_out),
            nn.Tanh()
        )

    def forward(self, x, t):
        x0 = self.hidden0(x, t)
        x0 = self.hidden1(x0)
        x0 = self.hidden2(x0)
        u, v = self.out(x0)
        return u, v
#TODO Do not hardcode the sizes of the hidden layers and the dropouts. They should be hyperparameters
#TODO Fix this forward pass. Goal is to break the network into two functions

#Generator network
#TODO Make a conditional generator
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features =  2  
        n_out = 2
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(100, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

#Adding some noise
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 2))
    return n

#Define a squaring function
def square(f):
    return lambda x: pow(f(x),2)

def ones_target(size):
    data  = Variable(torch.ones(size,1))
    return data

def zeros_target(size):
    data = Variable(torch.zeros(size,1))
    return data


#Define probability
def prob(self, coords, DiscriminatorNet):
    discriminator = self.DiscriminatorNet(coords)
    x , t = np.split(coords, 2)
    u , v = discriminator
    prob = integrate.quad(square(u) + square(v), -np.inf, np.inf, args=(t))
    return prob

#Implement Schrondinger's equation:
def schrodinger_loss(self, DiscriminatorNet, Potential, coords):
    discriminator = self.DiscriminatorNet(coords)
    potential = self.Potential(coords)
    u , v = discriminator
    schrodinger_loss = square(constants.hbar*grad(u,1) - grad(grad(v,0),0) - potential * v) + square(constants.hbar*grad(v,1) + grad(grad(u,0),0) + potential * u) 
    return schrondinger_loss 
#TODO Disable warning if the output is independent of the input

#Define loss 
def discriminator_loss(DiscriminatorNet, prob, coords, schrondinger_loss, real_data, fake_data, optimizer, a=0):
#Implement the smoothness of the wave function by Cauchy-Riemann equations. Disabling the CR and the prob functions for the time being and see what we get
    discriminator = self.DiscriminatorNet(coords)
    u, v = discriminator
    prob = self.prob(coords, discriminator)
    N =  real_data.size(0)
    loss = nn.BCEloss()
    schrondinger_loss_real = self.schrondinger_loss(discriminator, real_data, Potential =0)
    schrondinger_loss_fake = self.schrondinger_loss(discriminator, fake_data, Potential =0)
#Potential is set to 0 to see what comes out. Ecventually we will think of interesting functions for potential
    optimizer.zero_grad()
    pred_real = (square(u)(real_data) + square(v)(real_data))/2
    error_real = a*square(grad(u,0)-grad(v,1))(real_data) + a*square(grad(u,1)+grad(v,0))(real_data) + a*square(prob - 1)(real_data, discriminator) + schrondinger_loss_real(real_data) + loss(pred_real, ones_target(N))
    error_real.backward()

    pred_fake = square(u)(fake_data) + square(v)(fake_data) 
    error_fake = a*square(grad(u,0)-grad(v,1))(fake_data)+a*square(grad(u,1)+grad(v,0))(fake_data) + a*square(prob - 1)(fake_data, discriminator)+schrondinger_loss_fake(fake_data) + loss(pred_fake, zeros_target(N))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, pred_real, pred_fake
#BCEloss is designed to make real data equal to 1 and fake data equal to 0
#TODO make u**2 + v**2 equal to 1

def generator_loss(DiscriminatorNet, optimizer, fake_data):     
    discriminator = self.GeneratorNet()
    u, v = discriminator
    N = fake_data.size(0)
    optimizer.zero_grad()
    loss = nn.BCEloss()
    pred = square(u)(fake_data) + square(v)(fake_data)
    error = loss(pred, ones_target(N))
    error.backward()
    optimizer.step()
    return error 
