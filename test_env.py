import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import matplotlib.pyplot as plt
import time

from distributions import Categorical, DiagGaussian
from collections import namedtuple

import img_env

import utils

import model

from PIL import Image

from random import randint
import numpy as np

import argparse, os, logging, pickle

NUM_STEPS = 5
NUM_LABELS = 2

env = img_env.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=10, num_labels=NUM_LABELS)

fig = plt.figure()
observation = env.reset()

for t in range(NUM_STEPS):
	actionS = np.array(
					[np.random.choice(range(4)), np.random.choice(range(NUM_LABELS))])
	action = actionS[0]
	ax = plt.subplot(1, NUM_STEPS, t+1)
	ax.set_title('s{t}, a{t}={action}'.format(**locals()))		
	ax.imshow(observation[0,:,:], cmap=plt.cm.gray)
	observation, reward, done, info = env.step(actionS)
	
plt.savefig('test_env')
plt.show()
