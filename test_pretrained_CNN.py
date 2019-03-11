import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import random
import matplotlib.pyplot as plt

from distributions import Categorical, DiagGaussian
from collections import namedtuple

import img_env28, img_env28_extend

import utils

import model_extend

from PIL import Image

from random import randint
import numpy as np

from pretrained_CNN import CNN_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def smoothing_average(x, factor=10):
	running_x = 0
	for i in range(len(x)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + x[i] * U
		x[i] = running_x
	return x


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



def optimize_myNet(net, curr_label, optimizer, BATCH_SIZE=128, optimize_clf=False):
	if len(memory) < BATCH_SIZE:
		return
	# print ('Optimizing')
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))


	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.uint8).to(device)

	non_final_next_states = torch.stack([s for s in batch.next_state \
		if s is not None]).to(device)
	# print ('non_final_next_states', non_final_next_states.shape)
	state_batch = torch.stack(batch.state).to(device)
	# print ('state_batch.size', state_batch.size)
	action_batch = torch.stack(batch.action).to(device)
	reward_batch = torch.cat(batch.reward).to(device)


	_, Q_values_batch, clf_proba_batch, _, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])

	state_action_values = Q_values_batch.gather(1, action_batch[:, 0].view(BATCH_SIZE,1))
	next_state_values = torch.zeros(BATCH_SIZE).to(device)

	_, next_Q_values_batch, _, _, _= target_net.act(inputs=non_final_next_states.float(),states=non_final_next_states, masks=non_final_next_states[1])

	next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()

	expected_state_action_values = (next_state_values * GAMMA) + reward_batch # Compute the expected Q values
	loss_dist = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	curr_label_batch = torch.cat(batch.curr_label).to(device)
	loss_clf = F.nll_loss(clf_proba_batch, curr_label_batch)

	total_loss = loss_dist + loss_clf

	optimizer.zero_grad()
	total_loss.backward()
	# for param in net.parameters():
	for param in filter(lambda p: p.requires_grad, net.parameters()):
		# print (param)
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

	return total_loss, loss_clf, loss_dist



if __name__ == '__main__':
	BATCH_SIZE = 128
	NUM_STEPS = 1
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	EPS = 0.05
	NUM_LABELS = 2
	WINDOW_SIZE = 28
	NUM_EPISODES = 1000
	TARGET_UPDATE = 10

	# env = img_env28.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	env = img_env28_extend.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)

	net = model_extend.myNet_with_CNNpretrained(\
		obs_shape=env.observation_space.shape, \
		action_space=env.action_space, dataset='mnist').to(device)

	target_net = model_extend.myNet_with_CNNpretrained(\
		obs_shape=env.observation_space.shape, \
		action_space=env.action_space, dataset='mnist').to(device)

	target_net.load_state_dict(net.state_dict())
	target_net.eval()
	memory = ReplayMemory(10000)
	total_rewards = {}
	episode_durations = {}
	loss_classification = {}
	# optimizer_clf = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
	optimizer_clf = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.5)

	for i_episode in range(NUM_EPISODES):
		print ('episode ', i_episode)
		total_reward_i = 0
		observation = env.reset()
		
		curr_label = env.curr_label.item()
		for t in range(NUM_STEPS): # allow 100 steps
	
			actionS, Q_values, clf_proba, action_log_probs, states = net.act(
				inputs=torch.from_numpy(observation).float().resize_(1, observation.shape[0], observation.shape[1], observation.shape[2]).to(device),
				states=observation, masks=observation[1])
			actionS = actionS.cpu().numpy()[0]
			class_pred = actionS[1]
			last_observation = observation
			rand = np.random.rand()
			if rand < EPS:
				actionS = np.array(
					[np.random.choice(range(4)), np.random.choice(range(NUM_LABELS))])
			action = actionS[0]
			observation, reward, done, info = env.step(actionS)
			
			total_reward_i = reward + GAMMA*total_reward_i
			memory.push(torch.from_numpy(last_observation), torch.from_numpy(actionS), \
				torch.from_numpy(observation), torch.tensor([reward]).float(), torch.tensor([curr_label]))
# 			print ('t = %i: action = %i, class = %i, class_pred = %i, reward = %f'%(t, action, curr_label, class_pred, reward))
			optimize_myNet(net, curr_label, optimizer_clf, BATCH_SIZE)

			if done:
# 				# print ('Done after %i steps'%(t+1))
				break
	
		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(net.state_dict())
			
		loss_classification_i = F.nll_loss(clf_proba, env.curr_label.unsqueeze_(dim=0).to(device))
		if curr_label in total_rewards.keys():
			total_rewards[curr_label].append(total_reward_i)
			episode_durations[curr_label].append(t)
			loss_classification[curr_label].append(loss_classification_i)
		else:
			total_rewards[curr_label] = [total_reward_i]
			episode_durations[curr_label] = [t]
			loss_classification[curr_label] = [loss_classification_i]
	plt.title('Class 0')
	plt.subplot(3, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_durations[0], dtype=torch.float)
	plt.plot(smoothing_average(durations_t.numpy()))

	plt.subplot(3, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_rewards[0], dtype=torch.float)
	plt.plot(smoothing_average(total_rewards_t.numpy()))
	
	plt.subplot(3, 1, 3)
	plt.ylim(top=1)
	plt.xlabel('Episode')
	plt.ylabel('Loss Classification')
	loss_classification_t = torch.tensor(loss_classification[0], dtype=torch.float)
	plt.plot(smoothing_average(loss_classification_t.numpy()))
	plt.savefig('pretrained_CNN/pretrainedCNN_noNavigation_extend')
	plt.show()




	# CNN_input = torch.tensor(obs[0]).float().resize_(1,1,28,28)
	# output = base(CNN_input)

