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
import copy






# display_img = (np.reshape(env.curr_img.numpy(), (28, 28)) * 255).astype(np.uint8)
# img = Image.fromarray(display_img, 'L')
# # img.save('my.png')
# img.show()

# im0 = env.curr_img.numpy()


# display_img = observation[1,:,:]*255
# img = Image.fromarray(np.uint8(display_img), 'L')
# img.save('my_start.png')
# img.show()

def smoothing_average(x, factor=10):
    running_x = 0
    for i in range(len(x)):
        U = 1. / min(i+1, factor)
        running_x = running_x * (1 - U) + x[i] * U
        x[i] = running_x
    return x


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
			self.clf = Categorical(self.base.output_size, 2)#10)

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


def optimize_myNet(net, curr_label, BATCH_SIZE=128, optimize_clf=False):
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


	_, _, Q_values_batch, clf_proba_batch, _, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])

	# print (action_batch.shape)
	state_action_values = Q_values_batch.gather(1, action_batch[:, 0].view(BATCH_SIZE,1))
	# actual Q values = Q values indexed by sampled action
	next_state_values = torch.zeros(BATCH_SIZE).to(device)

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
	# print ('loss_clf', loss_clf.item())
	# print ('loss_dist', loss_dist.item())
	return total_loss.item(), loss_clf.item(), loss_dist.item()



if __name__ == '__main__':

	t0 = time.time()
	ts = ''#int(time.time())

	################# set logdir, required by lab GPU setting
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, required=True)
	args = parser.parse_args()
	assert os.path.exists(args.logdir), "Log directory non existant..."
	logging.basicConfig(filename=os.path.join(args.logdir, 'logs{ts}.txt'.format(**locals())),level=logging.DEBUG, \
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
	EPS = 0.05
	NUM_LABELS = 2

	env = img_env.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=10, num_labels=NUM_LABELS)
	num_episodes = 1000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	net = myNet(\
		obs_shape=env.observation_space.shape, \
		action_space=env.action_space, dataset='mnist').to(device)
	memory = ReplayMemory(10000)


	# create a standalone file to summarize all exp info and results
	results_file = {} 
	results_file['exp_info'] = {}
	results_file['exp_results'] = {}
	exp_name='bs{BATCH_SIZE}_steps{NUM_STEPS}_mnist{ts}_2cls_checkEndOfEpi'.format(**locals())
	results_file['exp_info']['name'] = exp_name
	results_file['exp_info']['num_episodes'] = num_episodes
	results_file['exp_info']['num_steps'] = NUM_STEPS
	results_file['exp_info']['device'] = device.type
	results_file['exp_info']['architecture'] = net.__repr__()

	action_code = {1:'D', 0: 'U', 2:'L', 3:'R'}

	for i_episode in range(num_episodes):
		results_file['exp_results']['episode_%i'%i_episode] = {}
		total_reward_i = 0
		observation = env.reset()
		curr_label_i = env.curr_label.item()
		actionS_i = []
		location_i = copy.deepcopy(observation[0])

		img_i = copy.deepcopy(observation[1])

		
		for t in range(NUM_STEPS):
			value, actionS, Q_values, clf_proba, action_log_probs, states = net.act(
				inputs=torch.from_numpy(observation).float().resize_(1, observation.shape[0], observation.shape[1], observation.shape[2]).to(device),
				states=observation, masks=observation[1])
			actionS = actionS.cpu().numpy()[0]
			class_pred = actionS[1]
			last_observation = observation
			rand = np.random.rand()
			if rand < EPS:
				actionS = np.array(
					[np.random.choice(range(4)), np.random.choice(range(NUM_LABELS))])
			actionS_i.append(actionS)
			action = actionS[0]
			# print(t, action_code[action], env.pos)
			# plt.imshow(observation[0], cmap=plt.cm.gray)
			# plt.show()
			observation, reward, done, info = env.step(actionS)
			total_reward_i += reward
			# location_i.append(observation[0])
			# img_i.append(observation[1])
			location_i = np.vstack((location_i, observation[0]))
			img_i = np.vstack((img_i, observation[1]))

			memory.push(
				torch.from_numpy(last_observation).to(device),
				torch.from_numpy(actionS).to(device),
				torch.from_numpy(observation).to(device),
				torch.tensor([reward]).float().to(device),
				torch.tensor([curr_label_i]).to(device))
			
			try: 
				total_loss_i, loss_clf_i, loss_dist_i = optimize_myNet(net, curr_label_i, BATCH_SIZE)
			except Exception: # no enough exp dans le memo
				optimize_myNet(net, curr_label_i, BATCH_SIZE)
			if done:
				# print ('Done after %i steps'%(t+1))
				break

		location_i = np.reshape(location_i, (NUM_STEPS+1,32,32))
		img_i = np.reshape(img_i, (NUM_STEPS+1, 32,32))
		results_file['exp_results']['episode_%i'%i_episode]['location'] = location_i
		results_file['exp_results']['episode_%i'%i_episode]['img'] = img_i
		results_file['exp_results']['episode_%i'%i_episode]['curr_label'] = curr_label_i
		results_file['exp_results']['episode_%i'%i_episode]['actionS']=actionS_i
		results_file['exp_results']['episode_%i'%i_episode]['total_reward']=total_reward_i
		results_file['exp_results']['episode_%i'%i_episode]['episode_duration']=t
		try:
			results_file['exp_results']['episode_%i'%i_episode]['loss_clf']=loss_clf_i
			results_file['exp_results']['episode_%i'%i_episode]['loss_dist']=loss_dist_i
			results_file['exp_results']['episode_%i'%i_episode]['total_loss']=total_loss_i
		except Exception:
			pass
		
	total_run_time = time.time()-t0
	logging.info('Experiments completed in %f seconds'%total_run_time)
	results_file['exp_info']['Total_Run_Time'] = total_run_time


	with open(os.path.join(result_dir, 'results_file{exp_name}'.format(**locals())), 'wb') as handle:
		pickle.dump(results_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

################################### PLOTS ####################################	

	episode_duration_cls0 = []
	total_reward_cls0 = []
	actionS_cls0 = []
	location_cls0 = []
	img_cls0 = []
	loss_clf_cls0 = []
	loss_dist_cls0 = []
	total_loss_cls0 = []

	for epi in range(results_file['exp_info']['num_episodes']):
		if results_file['exp_results']['episode_%i'%epi]['curr_label'] == 0:
			episode_duration_cls0.append(results_file['exp_results']['episode_%i'%epi]['episode_duration'])
			total_reward_cls0.append(results_file['exp_results']['episode_%i'%epi]['total_reward'])
			actionS_cls0.append(results_file['exp_results']['episode_%i'%epi]['actionS'])
			location_cls0.append(results_file['exp_results']['episode_%i'%epi]['location'])
			img_cls0.append(results_file['exp_results']['episode_%i'%epi]['img'])
			try:
				loss_clf_cls0.append(results_file['exp_results']['episode_%i'%epi]['loss_clf'])
				loss_dist_cls0.append(results_file['exp_results']['episode_%i'%epi]['loss_dist'])
				total_loss_cls0.append(results_file['exp_results']['episode_%i'%epi]['total_loss'])
			except Exception:
				pass

	fig1 = plt.figure() 
	plt.subplot(2, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_duration_cls0, dtype=torch.float)
	plt.plot(smoothing_average(durations_t.numpy()))

	plt.subplot(2, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_reward_cls0, dtype=torch.float)
	plt.plot(smoothing_average(total_rewards_t.numpy()))
	fig1.suptitle('Class 0')
	plt.savefig(os.path.join(result_dir, '{exp_name}'.format(**locals())))



	fig2 = plt.figure() 
	plt.subplot(3, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('loss_clf')
	loss_clf_t = torch.tensor(loss_clf_cls0, dtype=torch.float)
	plt.plot(smoothing_average(loss_clf_t.numpy()))

	plt.subplot(3, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('loss_dist')
	loss_dict_t = torch.tensor(loss_dist_cls0, dtype=torch.float)
	plt.plot(smoothing_average(loss_dict_t.numpy()))

	plt.subplot(3, 1, 3)
	plt.xlabel('Episode')
	plt.ylabel('total_loss')
	total_loss_t = torch.tensor(total_loss_cls0, dtype=torch.float)
	plt.plot(smoothing_average(total_loss_t.numpy()))
	fig2.suptitle('Class 0')
	plt.savefig(os.path.join(result_dir, 'loss_{exp_name}'.format(**locals())))




