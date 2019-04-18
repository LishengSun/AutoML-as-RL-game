import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import json
import pdb
import copy
import optparse

import random
import matplotlib.pyplot as plt

from distributions import Categorical, DiagGaussian
from collections import namedtuple

import img_env28, img_env28_jump

import utils
import seaborn as sns


import model_extend

from PIL import Image

from random import randint
import numpy as np
import os, time

from pretrained_CNN import CNN_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def smoothing_average(x, factor=10):
	running_x = 0
	X = copy.deepcopy(x)
	for i in range(len(X)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + X[i] * U
		X[i] = running_x
	return X


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
	print ('Optimizing')
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


	_, Q_values_batch, clf_proba_batch, log_clf_proba_batch, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])
	
	
	state_action_values = Q_values_batch.gather(1, action_batch[:,-1].view(BATCH_SIZE,1))
	next_state_values = torch.zeros(BATCH_SIZE).to(device)
	torch.zeros(BATCH_SIZE).to(device)
	_, next_Q_values_batch, _, _, _= target_net.act(inputs=non_final_next_states.float(),states=non_final_next_states, masks=non_final_next_states[1])
	

	next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()
	
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch # Compute the expected Q values


	loss_dist = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
	
	curr_label_batch = torch.cat(batch.curr_label).to(device)
	loss_clf = F.nll_loss(log_clf_proba_batch, curr_label_batch)

	total_loss = loss_dist + loss_clf

	optimizer.zero_grad()
	total_loss.backward()
	for name, param in net.named_parameters():
		print(name, torch.median(torch.abs(param.grad)).data if param.grad is not None else None)
	for param in filter(lambda p: p.requires_grad, net.parameters()):
		param.grad.data.clamp_(-1, 1) #gradient clipping prevent only the exploding gradient problem


	
	optimizer.step()

	return total_loss, loss_clf, loss_dist



if __name__ == '__main__':
	t0 = time.time()

	# Collect arguments 
	parser = optparse.OptionParser()

	parser.add_option('-e', '--NUM_EPISODES',
    	action="store", dest="NUM_EPISODES", type=int,
    	help="num of episodes to train", default=10000)

	parser.add_option('-s', '--NUM_STEPS',
    	action="store", dest="NUM_STEPS", type=int,
    	help="max num of steps for each episode", default=10)

	parser.add_option('-l', '--NUM_LABELS',
    	action="store", dest="NUM_LABELS", type=int,
    	help="num of labels shown to agent", default=2)

	parser.add_option('-w', '--WINDOW_SIZE',
    	action="store", dest="WINDOW_SIZE", type=int,
    	help="window size of each reveal", default=8)

	parser.add_option("-d", '--DEFREEZE_CNN', action="store_false", \
		dest="FREEZE_CNN")

	parser.add_option("-c", '--COMPLEX_NAVIGATION', action="store_true", \
		dest="COMPLEX_NAVIGATION")
	parser.set_defaults(FREEZE_CNN=True)
	parser.set_defaults(COMPLEX_NAVIGATION=False)

	options, args = parser.parse_args()

	NUM_STEPS = options.NUM_STEPS
	NUM_LABELS = options.NUM_LABELS
	WINDOW_SIZE = options.WINDOW_SIZE
	NUM_EPISODES = options.NUM_EPISODES
	FREEZE_CNN = options.FREEZE_CNN
	COMPLEX_NAVIGATION = options.COMPLEX_NAVIGATION

	BATCH_SIZE = 128
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	EPS = 0.05
	TARGET_UPDATE = 10
	RUNS = 1
	RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results/RL-agent')


	# env = img_env28.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	env = img_env28_jump.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)


	run_durations = []
	run_total_rewards = []
	run_loss_clf = []
	run_loss_dist = []
	run_loss_total = []
	run_observations = []
	run_actions = []
	run_labels = []

	for run in range(RUNS):
		net = model_extend.myNet_with_CNNpretrained(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, freeze_CNN=FREEZE_CNN, complex_navigation=COMPLEX_NAVIGATION, num_labels = NUM_LABELS, dataset='mnist').to(device)

		target_net = model_extend.myNet_with_CNNpretrained(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, freeze_CNN=FREEZE_CNN, complex_navigation=COMPLEX_NAVIGATION, num_labels = NUM_LABELS, dataset='mnist').to(device)

		target_net.load_state_dict(net.state_dict())
		target_net.eval()
		memory = ReplayMemory(10000)
		total_rewards = []
		episode_durations = []
		# loss_classification = []
		loss_clf = []
		loss_dist = []
		loss_total = []
		q_value = []
		observations = []
		actions = []
		labels = []
		# optimizer_clf = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
		optimizer_clf = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.5)
		# observation = env.reset()
		# print ('curr_label', env.curr_label)
		for i_episode in range(NUM_EPISODES):
			print ('run %i, episode %i'%(run, i_episode))
			total_reward_i = 0
			observations_i = []
			actions_i = []
			observation = env.reset()
			# observation = env.reset(NEXT=False)	#keep same image
			# print ('curr_label', env.curr_label)
			# plt.imshow(env.curr_img[0,:,:])
			# plt.show()
			curr_label = env.curr_label.item()
			labels.append(curr_label)

			for t in range(NUM_STEPS): # allow 100 steps
				observations_i.append(copy.deepcopy(observation))

				actionS, Q_value, clf_proba, log_clf_proba, states = net.act(
					inputs=torch.from_numpy(observation).float().resize_(1, observation.shape[0], observation.shape[1], observation.shape[2]).to(device),
					states=observation, masks=observation[1])
				
				actionS = actionS.cpu().numpy()[0]
				actions_i.append(copy.deepcopy(actionS))
				last_observation = observation
				rand = np.random.rand()
				if rand < EPS:
					actionS = np.array(
						[np.random.choice(range(28)), np.random.choice(range(28)), np.random.choice(range(2)), \
						np.random.choice(range(NUM_LABELS)), np.random.choice(range(28*28*2))])
				action_row, action_col, action_done, class_pred, action_squeeze = actionS
				observation, reward, done, info = env.step(actionS, clf_proba.detach().numpy()[0])

				total_reward_i = reward + GAMMA*total_reward_i
				memory.push(torch.from_numpy(last_observation), torch.from_numpy(actionS), \
					torch.from_numpy(observation), torch.tensor([reward]).float(), torch.tensor([curr_label]))
				# print ('t = %i: action = %i, class = %i, class_pred = %i, reward = %f, total_reward = %f'%(t, action, curr_label, class_pred, reward, total_reward_i))
				try:
					loss_total_i, loss_clf_i, loss_dist_i = optimize_myNet(net, curr_label, optimizer_clf, BATCH_SIZE)
					loss_total_i = loss_total_i.item() 
					loss_clf_i = loss_clf_i.item()
					loss_dist_i = loss_dist_i.item()
				except: # not enough examples in ReplayMemo

					loss_total_i, loss_clf_i, loss_dist_i = 9999, 9999, 9999

				if done:
					print ('%i steps, total_reward_i = %f, curr_label=%i, pred_label=%i'%(t+1, total_reward_i, curr_label, class_pred))
					break
		
			# Update the target network, copying all weights and biases in DQN
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(net.state_dict())
				
			# print (clf_proba, env.curr_label.unsqueeze_(dim=0).to(device))
			loss_classification_i = F.nll_loss(log_clf_proba, env.curr_label.unsqueeze_(dim=0).to(device))

			total_rewards.append(total_reward_i)
			episode_durations.append(t)
			# loss_classification.append(loss_classification_i.item())
			loss_clf.append(loss_clf_i)
			loss_dist.append(loss_dist_i)
			loss_total.append(loss_total_i)
			q_value.append(Q_value)
			observations.append(observations_i)
			actions.append(actions_i)


		run_durations.append(episode_durations)
		run_total_rewards.append(total_rewards)
		# run_loss_clf.append(loss_classification)
		run_loss_clf.append(loss_clf)
		run_loss_dist.append(loss_dist)
		run_loss_total.append(loss_total)
		run_observations.append(observations)
		run_actions.append(actions)
		run_labels.append(labels)


	# save everything for later analysis

	# torch.save(net.state_dict(), os.path.join(RESULT_DIR,'model_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.pth'.format(**locals())))
	# torch.save(optimizer_clf.state_dict(), os.path.join(RESULT_DIR,'optimizer_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.pth'.format(**locals())))


	
	# with open(os.path.join(RESULT_DIR,'run_durations_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.json'.format(**locals())), 'w') as outfile1:
	# 	json.dump(run_durations, outfile1)

	# with open(os.path.join(RESULT_DIR,'run_total_rewards_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.json'.format(**locals())), 'w') as outfile2:
	# 	json.dump(run_total_rewards, outfile2)

	# with open(os.path.join(RESULT_DIR,'run_loss_clf_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.json'.format(**locals())), 'w') as outfile3:
	# 	json.dump(run_loss_clf, outfile3)


	# with open(os.path.join(RESULT_DIR,'run_labels_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10.json'.format(**locals())), 'w') as outfile4:
	# 	json.dump(run_labels, outfile4)

	# np.save(os.path.join(RESULT_DIR,'run_observations_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10'.format(**locals())), run_observations)
		
	# np.save(os.path.join(RESULT_DIR,'run_actions_freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10'.format(**locals())), run_actions)
		

	print ('total runtime = %f sec.'%(time.time()-t0))


	plt.subplot(3, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_durations[0], dtype=torch.float)
	# plt.plot(smoothing_average(durations_t.numpy()))
	sns.tsplot(data=[smoothing_average(run_durations[i]) for i in range(len(run_durations))], \
		time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 1), color='red')

	plt.subplot(3, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_rewards[0], dtype=torch.float)
	# plt.plot(smoothing_average(total_rewards_t.numpy()))
	sns.tsplot(data=[smoothing_average(run_total_rewards[i]) for i in range(len(run_total_rewards))], \
		time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')

	# sns.tsplot(data=run_total_rewards, time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')

	plt.subplot(3, 1, 3)
	plt.ylim(top=5)
	plt.xlabel('Episode')
	plt.ylabel('Loss Navigation')
	# loss_classification_t = torch.tensor(loss_classification[0], dtype=torch.float)
	# plt.plot(smoothing_average(loss_classification_t.numpy()))
	

	l1 = sns.tsplot(data=[smoothing_average(run_loss_dist[i][50:], factor=100) for i in range(len(run_loss_dist))], \
		time=list(range(50, NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 3), color='red', condition='navig')
	l2 = sns.tsplot(data=[smoothing_average(run_loss_total[i][50:], factor=100) for i in range(len(run_loss_dist))], \
		time=list(range(50, NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 3), color='green', condition='total')
	l3 = sns.tsplot(data=[smoothing_average(run_loss_clf[i][50:], factor=100) for i in range(len(run_loss_dist))], \
		time=list(range(50, NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 3), color='blue', condition='clf')
	# sns.tsplot(data=run_loss_clf, time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')
	# plt.legend([l1, l2, l3], ['dist', 'total', 'clf'])
	plt.savefig(os.path.join(RESULT_DIR,'freeze{FREEZE_CNN}_complexNavig{COMPLEX_NAVIGATION}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws_rw-10'.format(**locals())))
	plt.show()

	




	
