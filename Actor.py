import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# env = gym.make('CartPole-v0').unwrapped

class A3CActor(nn.module):

    def __init__(self, num_inputs, num_outputs, hidden=64):
        super(A3CActor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.action_mean = nn.Linear(hidden, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
