import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import random
import pdb


from distributions import Categorical, DiagGaussian, Raw_and_Categorical
from collections import namedtuple

import img_env28_jump

import utils

import model_extend

from PIL import Image

from random import randint
import numpy as np

from pretrained_CNN import CNN_pretrained





class CNNpretrained_Base(nn.Module):
	def __init__(self, freeze_CNN=True):
		super(CNNpretrained_Base, self).__init__()
		# load the pretrained CNN
		CNN_pretr = CNN_pretrained()
		CNN_state_dict = torch.load('./pretrained_CNN/results/model.pth')
		CNN_pretr.load_state_dict(CNN_state_dict)
		

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

class myClassifier_with_CNNpretrained(nn.Module):
	""" 
		Blackbox Classifier 
	"""
	def __init__(self):
		super(myClassifier_with_CNNpretrained, self).__init__()
		# load the pretrained CNN
		self.CNN_pretr = CNN_pretrained()
		CNN_state_dict = torch.load('./pretrained_CNN/results/model.pth')
		self.CNN_pretr.load_state_dict(CNN_state_dict)
		# print ('freeze ittttttttt')
		for param in self.CNN_pretr.parameters(): 
				param.requires_grad = False

	def forward(self, inputs):
		log_softmax, softmax = self.CNN_pretr(inputs)
		

		return log_softmax, softmax

	





class myNet_with_CNNpretrained(nn.Module):
	"""
		Contained 2 parts: 
		(1) Blackbox classifier (frozen by default, defreeze by setting Freeze_CNN=False), 
		(2) Navigation = pretrained CNN + flexible navigation softmax
	"""
	def __init__(self, obs_shape, action_space, freeze_CNN=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet_with_CNNpretrained, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = CNNpretrained_Base(freeze_CNN)

		else:
			raise NotImplementedError

		if action_space.__class__.__name__=='Discrete':
			self.num_outputs = action_space.n
			self.dist = Raw_and_Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], self.num_outputs)
			# self.num_outputs_row = action_space[0].n
			# self.num_outputs_col = action_space[1].n
			# self.num_outputs_done = action_space[2].n
			# self.dist = Categorical(num_outputs_row+num_outputs_col+num_outputs_done, action_space[0].n*action_space[1].n*action_space[2].n)


		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			num_labels = num_labels
			if freeze_CNN: # treat clf CNN as a blackbox 
				self.clf = myClassifier_with_CNNpretrained() 
			else: # defreeze the blackbox CNN
				self.clf = Categorical(self.base.output_size, 10)
			
		self.state_size = self.base.state_size


	def forward(self, inputs, states, masks):
		raise NotImplementedError

	


	def act(self, freeze_CNN, inputs, states, masks, deterministic=False):
		actor_features = self.base(inputs[:,1:2,:,:])# only takes img as input to blackbox clf
		
		actor_features_with_location = torch.cat((actor_features, inputs[:,0,:,:].view(inputs.size(0), -1)), 1) # concat the location image
	
		raw_dist, dist = self.dist(actor_features_with_location)
		Q_value = raw_dist
		
		
		if deterministic:
			action = dist.mode()
			
		else:
			action = dist.sample()
			

		if self.dataset in img_env28_jump.IMG_ENVS:
			if freeze_CNN:
				log_clf_softmax, clf_softmax = self.clf(inputs[:,1:2,:,:])# feed image directly to the blackbox CNN
			else:
				# pdb.set_trace()
				log_clf_softmax, clf_softmax = self.clf(actor_features).logits,  self.clf(actor_features).probs# pass through self.base

			classif = clf_softmax.data.max(1, keepdim=True)[-1]#.item()

			
			action = torch.cat([action, classif], 1)

		return action, Q_value, clf_softmax, log_clf_softmax, states#dist.logits = Q values













