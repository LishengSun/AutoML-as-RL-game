import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import random

from distributions import Categorical, DiagGaussian
from collections import namedtuple

import img_env28 

import utils

import model_extend

from PIL import Image

from random import randint
import numpy as np

from pretrained_CNN import CNN_pretrained





class CNNpretrained_Base(nn.Module):
	def __init__(self):
		super(CNNpretrained_Base, self).__init__()
		# load the pretrained CNN
		CNN_pretr = CNN_pretrained()
		CNN_state_dict = torch.load('./pretrained_CNN/results/model.pth')
		CNN_pretr.load_state_dict(CNN_state_dict)
		# freeze it
		for param in CNN_pretr.parameters(): 
			param.requires_grad = False

		self.conv1 = CNN_pretr.conv1
		self.conv2 = CNN_pretr.conv2
		self.conv2_drop = CNN_pretr.conv2_drop
		self.fc1 = CNN_pretr.fc1

		# self.CNN = CNN_pretr
	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		self.features = x
		return x

	@property
	def state_size(self):
		if hasattr(self, 'gru'):
			return 50 # output size of pretrained CNN (last-1 layer)
		else:
			return 1

	@property
	def output_size(self):
		return 50




class myNet_with_CNNpretrained(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet_with_CNNpretrained, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = CNNpretrained_Base()

		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Discrete": # our case
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs)
		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			num_labels = 10
			self.clf = Categorical(self.base.output_size, num_labels)

		self.state_size = self.base.state_size


	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		actor_features = self.base(inputs[:,1:2,:,:])# only takes img as input to the pretrained CNN
		# print ('actor_features.shape ', actor_features.shape)
		# print ('inputs[:,0,:,:]', inputs[:,0,:,:].shape)
		actor_features_with_location = torch.cat((actor_features, inputs[:,0,:,:].view(inputs.size(0), -1)), 1)
		# print ('actor_features_with_location', actor_features_with_location.shape)
		dist = self.dist(actor_features_with_location)
#         print (dist)
		Q_values = dist.logits
		
		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)

		if self.dataset in img_env28.IMG_ENVS:
			clf = self.clf(actor_features)
			clf_proba = clf.logits
			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()
			
			action = torch.cat([action, classif], 1)
			# print ('action', action)
			# print ('classif', classif)
			action_log_probs += clf_proba.gather(1, classif)

		return action, Q_values, clf_proba, action_log_probs, states #dist.logits = Q values
