###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
#
#
# RBM module
#
# 2017 Guido Cossu <gcossu.work@gmail.com>
#
##############################################################


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from tqdm import *

import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp


class CSV_Ising_dataset(Dataset):
    """ Defines a CSV reader """
    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)


class RBM(nn.Module):
    """Defines a torch module implementing a Restricted Boltzmann Machine.
    """

    def __init__(self, n_vis=784, n_hid=500, k=5, enable_cuda=False):
        """Initialises the module
        Args:
            n_vis: Number of visible nodes in the RBM.
            n_hid: Number of hidden nodes.
            k: Number of constrative divergence steps.
            enable_cuda: Boolean defining whether cuda tensor should be used

        Attributes:
            n_vis: Number of visible nodes in the RBM.
            n_vis: Number of visible nodes in the RBM.
            n_hid: Number of hidden nodes.
            CDiter: Number of constrative divergence steps.
            enable_cuda: Boolean defining whether cuda tensor should be used
            dtype: datatype of the tensors used in the model.
        """

        super(RBM, self).__init__()
        """Defines the constructor for the class.
        """
        # Set attributes, 
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.CDiter = k
        self.enable_cuda = enable_cuda

        # Enable or disable cuda
        if self.enable_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        # Initialise weights with a gaussian, and biases with zeros
        self.W = nn.Parameter(torch.zeros(n_hid, n_vis).type(self.dtype))
        nn.init.normal(self.W, mean=0, std=0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis).type(self.dtype))
        self.h_bias = nn.Parameter(torch.zeros(n_hid).type(self.dtype))

    def sample_probability(self, prob, random):
        """Get samples from a tensor of probabilities.
        
        Args:
            probs: Tensor of probabilities
            rand: Tensor (of the same shape as probs) of random values
            random: Binary sample of probabilities
        """
        return F.relu(torch.sign(prob - random))


    def hidden_from_visible(self, visible):
        """Samples the hidden (latent) variables given the visible.
        
        Args:
            visible: Tensor containing the states of the visible nodes

        Returns:
            new_states: Tensor containing binary (1 or 0) states of the hidden variables
            probability: Tensor containing probabilities P(H_i = 1| {V})
        """

        # Calculates the probabilities of each node being equal to one, and generates a
        # tensor containing random numbers between 0 and 1.
        probability = torch.sigmoid(F.linear(visible, self.W, self.h_bias))
        random_field = Variable(torch.rand(probability.size()).type(self.dtype))

        # Compares the values of the probability and random numbers and returns a tensor
        # of ones and zeros accordingly.
        new_states = self.sample_probability(probability, random_field)
        return new_states, probability

    def visible_from_hidden(self, hid):
        """Samples the hidden (latent) variables given the visible.
        
        Args:
            hid: Tensor containing the states of the hidden nodes

        Returns:
            new_states: Tensor containing binary (1 or 0) states of the visible variables
            probability: Tensor containing probabilities P(V_i = 1| {H})
        """

        # Calculates the probabilities of each node being equal to one, and generates a
        # tensor containing random numbers between 0 and 1.
        probability = torch.sigmoid(F.linear(hid, self.W.t(), self.v_bias))
        random_field = Variable(torch.rand(probability.size()).type(self.dtype))

        # Compares the values of the probability and random numbers and returns a tensor
        # of ones and zeros accordingly.
        new_states = self.sample_probability(probability, random_field)
        return new_states, probability

    def new_state(self, visible, use_prob=False):
        """Implements one steps of contrastive divergence.

        Args:
            visible: Tensor containing current state of visible nodes

        Returns:
            hidden: Tensor containing binary (1 or 0) states of the hidden variables
            probhid: Tensor containing probabilities P(H_i = 1| {V})
            new_visible: Tensor containing binary (1 or 0) states of the visible variables
            probvis: Tensor containing probabilities P(V_i = 1| {H})
        """
        hidden, probhid = self.hidden_from_visible(visible)
        if (use_prob):
            new_visible, probvis = self.visible_from_hidden(probhid)
        else:
            new_visible, probvis = self.visible_from_hidden(hidden)

        return hidden, probhid, new_visible, probvis

    def forward(self, input):
        """Necessary function for Module classes.
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        This implementation runs k contrastive divergence steps and returns the current
        states of the machine's visible and hidden variables.

        Returns:
            hidden: Tensor containing binary (1 or 0) states of the hidden variables
            h_prob: Tensor containing probabilities P(H_i = 1| {V})
            vis: Tensor containing binary (1 or 0) states of the visible variables
            v_prob: Tensor containing probabilities P(V_i = 1| {H})

        """
        # Runs 1 CD step
        hidden, h_prob, vis, v_prob = self.new_state(input)
        
        # Runs K-1 CD steps
        for _ in range(self.CDiter-1):
            hidden, h_prob, vis, v_prob = self.new_state(vis, use_prob=False)
        
        return vis, hidden, h_prob, v_prob

    def loss(self, ref, test):
        """Implements the loss function of the module, and defines it to be the 
        mean-squared error between the reference and the test value.

        Args:
            ref: Tensor of input data
            test: Tensor of state of visible nodes after k CD steps
        """
        return F.mse_loss(test, ref, size_average=True)

    def free_energy(self, v):
        """Computes log( p(v) ) as given by eq 2.20 in Asja Fischer's thesis.

        Args:
            v: Tensor containing the states of the visible nodes to calculate the
               free energy of.

        Returns:

        """
        vbias_term = v.mv(self.v_bias).double()  # = v*v_bias
        wx_b = F.linear(v, self.W, self.h_bias).double()  # = vW^T + h_bias
        
        # sum over the elements of the vector
        hidden_term = wx_b.exp().add(1).log().sum(1)  

        # notice that for batches of data the result is still a vector of size num_batches
        free_energy = -(hidden_term + vbias_term).mean() # mean along the batches

        return free_energy

    def backward(self, target, vk):
        """Backpropagation. Updates the gradients of the parameters according to the contrastive
        divergence algorithm.

        Args:
            vk: Tensor containing the states of the visible nodes after kCD steps
            target: Tensor containing training data for this train step.
        """
        # Calculates p(H_i | v) where v is the input data
        probability = torch.sigmoid(F.linear(target, self.W, self.h_bias))

        # Calculates p(H_i | v^(k)) 
        h_prob_negative = torch.sigmoid(F.linear(vk, self.W, self.h_bias))
        
        # Update the weights
        training_set_avg = probability.t().mm(target)
        
        # The minus sign comes from the implementation of the SGD in pytorch 
        # see http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        # the learning rate has a negative sign in front
        self.W.grad = -(training_set_avg - h_prob_negative.t().mm(vk)) / probability.size(0)

        # Update the biases
        self.v_bias.grad = -(target - vk).mean(0)
        self.h_bias.grad = -(probability - h_prob_negative).mean(0)
