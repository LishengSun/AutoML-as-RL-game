import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import random
import pdb


from distributions import Categorical, DiagGaussian
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
		
		if freeze_CNN: # freeze it
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

class myClassifier_with_CNNpretrained(nn.Module):
	def __init__(self, obs_shape, action_space, freeze_CNN=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myClassifier_with_CNNpretrained, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = CNNpretrained_Base(freeze_CNN)

		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			num_labels = num_labels
			self.clf = Categorical(self.base.output_size, num_labels)

	def forward(self, inputs, deterministic=False):
		actor_features = self.base(inputs[:,1:2,:,:])# only takes img as input to the pretrained CNN
		if self.dataset in img_env28_jump.IMG_ENVS:
			clf = self.clf(actor_features)
			
			clf_proba = clf.logits
			
			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()

		return classif, clf_proba
		





class myNet_with_CNNpretrained(nn.Module):
	def __init__(self, obs_shape, action_space, freeze_CNN=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet_with_CNNpretrained, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = CNNpretrained_Base(freeze_CNN)

		else:
			raise NotImplementedError

		if [action_space[i].__class__.__name__=='Discrete' for i in range(3)]: # our case: len(env.action_space)=3 channels of actions
			num_outputs_row = action_space[0].n
			num_outputs_col = action_space[1].n
			num_outputs_done = action_space[2].n
			self.dist_row = Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_row)
			self.dist_col = Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_col)
			self.dist_done = Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_done)
			self.dist = Categorical(num_outputs_row+num_outputs_col+num_outputs_done, 28*28*2)


		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			num_labels = num_labels
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
		dist_row = self.dist_row(actor_features_with_location)
		dist_col = self.dist_col(actor_features_with_location)
		dist_done = self.dist_done(actor_features_with_location)
		# print ('dist_row: ', dist_row)
		# print ('dist_row_logits: ', dist_row.logits)
		dist = self.dist(torch.cat([dist_row.logits, dist_col.logits, dist_done.logits], 1))
		# print ('dist: ', dist)
		Q_value = dist.logits
		# print ('Q value: ', Q_value)

		
		if deterministic:
			action_row = dist_row.mode()
			action_col = dist_col.mode()
			action_done = dist_done.mode()
		else:
			action_row = dist_row.sample()
			action_col = dist_col.sample()
			action_done = dist_done.sample()

		# action_row_log_probs = dist_row.log_probs(action_row)
		# action_col_log_probs = dist_row.log_probs(action_col)
		# action_done_log_probs = dist_row.log_probs(action_done)

		if self.dataset in img_env28_jump.IMG_ENVS:
			clf = self.clf(actor_features)
			clf_proba = clf.logits
			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()
			
			action = torch.cat([action_row, action_col, action_done, classif], 1)
			# print ('action_row', action_row)
			# print ('action_col', action_col)
			# print ('action_done', action_done)
			# print ('classif', classif)

		return action, Q_value, clf_proba, states #dist.logits = Q values
