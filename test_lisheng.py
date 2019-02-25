import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import matplotlib.pyplot as plt

from distributions import Categorical, DiagGaussian
from collections import namedtuple

import img_env 

import utils

import model

from PIL import Image

from random import randint
import numpy as np
import logging, argparse
import os
import pickle
import time




class myNet(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
		elif len(obs_shape) == 1:
			assert not recurrent_policy, \
				"Recurrent policy is not implemented for the MLP controller"
			self.base = MLPBase(obs_shape[0])
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Discrete": # our case
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			self.clf = Categorical(self.base.output_size, 10)

		self.state_size = self.base.state_size

	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		value, actor_features, states = self.base(inputs, states, masks)
		self.actor_features = actor_features
		dist = self.dist(actor_features) 
		Q_values = dist.logits
		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)
		if self.dataset in img_env.IMG_ENVS:
			clf = self.clf(self.actor_features)
			clf_proba = clf.logits
			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()
			action = torch.cat([action, classif], 1)
			action_log_probs += clf.log_probs(classif)

		return value, action, Q_values, clf_proba, action_log_probs, states #dist.logits = Q values







class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'curr_label'))


def optimize_myNet(net, curr_label, BATCH_SIZE=32, optimize_clf=False):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))


	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8).to(device)

	non_final_next_states = torch.stack([s for s in batch.next_state \
    	if s is not None])
	# print ('non_final_next_states', non_final_next_states.shape)
	state_batch = torch.stack(batch.state)
	# print ('state_batch.size', state_batch.size)
	action_batch = torch.stack(batch.action)
	reward_batch = torch.cat(batch.reward)
	

	_, _, Q_values_batch, clf_proba_batch, _, _ = net.act(inputs=state_batch.float(), \
			states=state_batch, masks=state_batch[1])

	# print (action_batch.shape)
	state_action_values = Q_values_batch.gather(1, action_batch[:, 0].view(BATCH_SIZE,1))
	# actual Q values = Q values indexed by sampled action
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	
	_, _, next_Q_values_batch, _, _, _= net.act(inputs=non_final_next_states.float(),states=non_final_next_states, masks=non_final_next_states[1])
	
	next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()

	expected_state_action_values = (next_state_values * GAMMA) + reward_batch # Compute the expected Q values
	loss_dist = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	curr_label_batch = torch.cat(batch.curr_label)
	loss_clf = F.nll_loss(clf_proba_batch, curr_label_batch)

	total_loss = loss_dist + loss_clf
	optimizer_dist = optim.RMSprop(net.parameters())
	optimizer_dist.zero_grad()
	total_loss.backward()
	for param in net.dist.parameters():
		param.grad.data.clamp_(-1, 1)
		# print (param.grad.data)
	optimizer_dist.step()

	# if optimize_clf:
	#
	# 	optimizer_clf = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
	# 	optimizer_clf.zero_grad()
	# 	loss_clf.backward()
	# 	for param in net.dist.parameters():
	# 		param.grad.data.clamp_(-1, 1)
	# 	optimizer_clf.step()
	return loss_clf, loss_dist




if __name__ == '__main__':

	t0 = time.time()
	
	################# set logdir, required by lab GPU setting
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, required=True)
	args = parser.parse_args()
	assert os.path.exists(args.logdir), "Log directory non existant..."
	logging.basicConfig(filename=os.path.join(args.logdir, 'logs.txt'),level=logging.DEBUG, \
		format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode='w')

	############### set result dir to save results and plots
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	logging.info('Current dir = %s'%curr_dir)
	result_dir = os.path.join(curr_dir, 'results')
	

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	logging.info('Result dir = %s'%result_dir)
	if torch.cuda.is_available(): 
		logging.info(os.environ["CUDA_VISIBLE_DEVICES"])
	else:
		logging.info('No GPU AVAILABLE')

    ################## conf net
	BATCH_SIZE = 128
	NUM_STEPS = 20
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length

	env = img_env.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=10)
	num_episodes = 5000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = myNet(obs_shape=env.observation_space.shape, action_space=env.action_space, dataset='mnist').to(device)
	memory = ReplayMemory(10000)

	experiment_info = {} # create a standalone file to summarize all exp info and results
	experiment_info['name'] = 'digits_check_EndOfEpisode'
	experiment_info['num_episodes'] = num_episodes
	experiment_info['NUM_STEPS'] = NUM_STEPS
	experiment_info['device'] = device.type
	experiment_info['architecture'] = net.__repr__()

	experiment_info['results'] = {}





	for i_episode in range(num_episodes):
		# print ('episode:', i_episode)
		experiment_info['results']['episode_%i'%i_episode] = {}
		total_reward_i = 0
		observation = env.reset()
		curr_label_i = env.curr_label.item()
		experiment_info['results']['episode_%i'%i_episode]['curr_label'] = curr_label_i
		action_trajectory_i = [observation]

		for t in range(NUM_STEPS):
			# print ('time step:', t)
			value, actionS, Q_values, clf_proba, action_log_probs, states = net.act(
				inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device),
				states=observation, masks=observation[1])
			if device.type == 'cuda':
				action = actionS.cpu().numpy()[0][0]
				class_pred = actionS.cpu().numpy()[0][1]
				actionS = actionS.cpu().numpy()[0]
			else:
				action = actionS.numpy()[0][0]
				class_pred = actionS.numpy()[0][1]
				actionS = actionS.numpy()[0]
			last_observation = observation
			observation, reward, done, info = env.step(actionS)
			# print ('done?', done)
			# print ('reward', reward)
			action_trajectory_i.append(observation)
			total_reward_i = reward + GAMMA*total_reward_i
			# print ('total_reward_i', total_reward_i)
			memory.push(
				torch.from_numpy(last_observation).to(device),
				torch.from_numpy(actionS).to(device),
				torch.from_numpy(observation).to(device),
				torch.tensor([reward]).float().to(device),
				torch.tensor([curr_label_i]).to(device))

			optimize_myNet(net, curr_label_i, BATCH_SIZE)

			if done:
				# print ('Done after %i steps'%(t+1))
				break

		experiment_info['results']['episode_%i'%i_episode]['action_trajectory']=action_trajectory_i
		experiment_info['results']['episode_%i'%i_episode]['total_reward']=total_reward_i
		experiment_info['results']['episode_%i'%i_episode]['episode_duration']=t

		
	total_run_time = time.time()-t0
	logging.info('Experiments completed in %f seconds'%total_run_time)
	experiment_info['Total_Run_Time'] = total_run_time

	with open(os.path.join(result_dir, 'experiment_info_%s.pickle'%experiment_info['name']), 'wb') as handle:
		pickle.dump(experiment_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

	

	episode_duration_cls0 = []
	total_reward_cls0 = []

	for epi in range(experiment_info['num_episodes']):
		# print (experiment_info['results']['episode_%i'%epi]['curr_label'])
		if experiment_info['results']['episode_%i'%epi]['curr_label'] == 4:
			episode_duration_cls0.append(experiment_info['results']['episode_%i'%epi]['episode_duration'])
			total_reward_cls0.append(experiment_info['results']['episode_%i'%epi]['total_reward'])


	plt.title('Class 0')
	plt.subplot(2, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_duration_cls0, dtype=torch.float)
	plt.plot(durations_t.numpy())

	plt.subplot(2, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_reward_cls0, dtype=torch.float)
	plt.plot(total_rewards_t.numpy())
	plt.savefig(os.path.join(result_dir, 'plot_%s.png'%experiment_info['name']))

