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
	def __init__(self):#, obs_shape, action_space, freeze_CNN=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myClassifier_with_CNNpretrained, self).__init__()
		# load the pretrained CNN
		self.CNN_pretr = CNN_pretrained()
		CNN_state_dict = torch.load('./pretrained_CNN/results/model.pth')
		self.CNN_pretr.load_state_dict(CNN_state_dict)
		# freeze ittttttttt
		for param in self.CNN_pretr.parameters(): 
				param.requires_grad = False

	def forward(self, inputs):
		log_softmax, softmax = self.CNN_pretr(inputs)
		

		return log_softmax, softmax

	# def __init__(self, obs_shape, action_space, freeze_CNN=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
	# 	super(myClassifier_with_CNNpretrained, self).__init__()
	# 	self.dataset = dataset
	# 	if len(obs_shape) == 3: #our mnist case
	# 		self.base = CNNpretrained_Base(freeze_CNN)

	# 	else:
	# 		raise NotImplementedError

	# 	if dataset in ['mnist', 'cifar10']:
	# 		num_labels = num_labels
	# 		self.clf = Categorical(self.base.output_size, num_labels)

	# def forward(self, inputs, deterministic=False):
	# 	print ('clfing')
	# 	actor_features = self.base(inputs[:,1:2,:,:])# only takes img as input to the pretrained CNN
	# 	if self.dataset in img_env28_jump.IMG_ENVS:
	# 		clf = self.clf(actor_features)
			
	# 		clf_logits = clf.logits
	# 		clf_proba = clf.probs
			
	# 		if deterministic:
	# 			classif = clf.mode()
	# 		else:
	# 			classif = clf.sample()

	# 	return classif, clf_logits, clf_proba
		





class myNet_with_CNNpretrained(nn.Module):
	def __init__(self, obs_shape, action_space, freeze_CNN=True, complex_navigation=True, num_labels = 10, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
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
			if complex_navigation:
				self.dist_row = FC_navigation_head(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_row)
				self.dist_col = FC_navigation_head(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_col)
				self.dist_done = FC_navigation_head(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_done)


			else:
				self.dist_row = Raw_and_Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_row)

				self.dist_col = Raw_and_Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_col)
				self.dist_done = Raw_and_Categorical(self.base.output_size+obs_shape[1]*obs_shape[2], num_outputs_done)

			self.dist = Categorical(num_outputs_row+num_outputs_col+num_outputs_done, 28*28*2)


		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			num_labels = num_labels
			# self.clf = Categorical(self.base.output_size, num_labels)
			self.clf = myClassifier_with_CNNpretrained() # treat CNN as a blackbox 
		self.state_size = self.base.state_size


	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		actor_features = self.base(inputs[:,1:2,:,:])# only takes img as input to the pretrained CNN
		# print ('actor_features.shape ', actor_features.shape)
		# print ('inputs[:,0,:,:]', inputs[:,0,:,:].shape)
		actor_features_with_location = torch.cat((F.relu(actor_features), inputs[:,0,:,:].view(inputs.size(0), -1)), 1)
		# print ('actor_features_with_location', actor_features_with_location.shape)
		raw_row, dist_row = self.dist_row(actor_features_with_location)
		raw_col, dist_col = self.dist_col(actor_features_with_location)
		raw_done, dist_done = self.dist_done(actor_features_with_location)
		# print ('raw_row', raw_row)
		# dist = self.dist(F.relu(torch.cat([dist_row.logits, dist_col.probs, dist_done.probs], 1)))
		dist = self.dist(torch.cat([F.relu(raw_row), F.relu(raw_col), F.relu(raw_done)], 1))
		Q_value = dist.logits

		# print ('Q value: ', Q_value)

		
		if deterministic:
			action_row = dist_row.mode()
			action_col = dist_col.mode()
			action_done = dist_done.mode()
			action_squeeze = dist.mode()
		else:
			action_row = dist_row.sample()
			action_col = dist_col.sample()
			action_done = dist_done.sample()
			action_squeeze = dist.sample()

		# action_row_log_probs = dist_row.log_probs(action_row)
		# action_col_log_probs = dist_row.log_probs(action_col)
		# action_done_log_probs = dist_row.log_probs(action_done)

		if self.dataset in img_env28_jump.IMG_ENVS:
			# clf = self.clf(actor_features)
			# clf_proba = clf.logits

			log_softmax, softmax = self.clf(inputs[:,1:2,:,:])# treat CNN as a blackbox and output it directly as part of stopping criterion
			classif = softmax.data.max(1, keepdim=True)[-1]#.item()

			# if deterministic:
			# 	# classif = clf.mode()
			# else:
			# 	classif = clf.sample()
			
			action = torch.cat([action_row, action_col, action_done, classif, action_squeeze], 1)


		return action, Q_value, softmax, log_softmax, states #dist.logits = Q values



class FC_navigation_head(nn.Module):
	def __init__(self, num_input, num_output):
		super(FC_navigation_head, self).__init__()
		self.fc1 = nn.Linear(num_input, 320)
		self.fc2 = nn.Linear(320, 160)
		self.fc3 = nn.Linear(160, 80)
		self.categorical = Categorical(80, num_output)

	def forward(self, x):
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.categorical(x)
		return x










