"""
Based on img_env28_extend.py, changed to have jump action space 
For this, we flatten the image (row-major) to 28*28=784 list, 
and let the action space to have 785 (extra action for done) discrete values.
Then when the agent chooses a action value among 0, ..., 784, we just recover it to have the exact location 
with index_row = action//28 and index_col = action%28
--Lisheng
"""


import numpy as np

import torch
from gym.spaces import Discrete, Box
import torchvision.transforms as T
from torchvision import datasets
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import pdb


CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Hellinger_distance(P, Q):
    """
    compute the H distance between 2 proba distributions P and Q
    P and Q tensors, of size (1, k)
    1 = number of examples
    k = len of each proba distribution

    """
    if Q.shape[1] == 1: # one-hot target
        target = Q.item()
        Q = torch.zeros((P.shape))
        Q[:, target] = 1
    H_distance = np.sum([(P[:,i]-Q[:,i])**2 for i in range(P.shape[1])]) 
    H_distance = np.sqrt(H_distance) / np.sqrt(2)
    return H_distance

def get_data_loader(env_id, train=True):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if env_id in IMG_ENVS:
        if env_id == 'mnist':
            transform = T.Compose([
                           # T.Resize(size=(32, 32)),
                           T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))
                       ])
            dataset = datasets.MNIST
        elif env_id == 'cifar10':
            dataset = datasets.CIFAR10
        elif env_id == 'cifar100':
            dataset = datasets.CIFAR100
        elif env_id == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            if train:
                data_dir = ''
            else:
                data_dir = ''
            dataset = datasets.ImageFolder(
                data_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        loader = torch.utils.data.DataLoader(
            dataset('data', train=train, download=True,
                    transform=transform),
            batch_size=1, shuffle=True, **kwargs)
    return loader


class ImgEnv(object):
    def __init__(self, dataset, train, max_steps, channels, window=5, num_labels=10):
        # Jump action space has 28/window_size*28/window_size+1 actions: 
        
        self.channels = channels
        self.data_loader = get_data_loader(dataset, train=train)
        self.window = window
        self.max_steps = max_steps
        self.num_labels = num_labels
        self.num_row_choices = math.ceil(28/self.window)
        self.num_col_choices = math.ceil(28/self.window)

        # self.action_space = [Discrete(self.num_row_choices), Discrete(self.num_col_choices), Discrete(2)]
        self.action_space = Discrete(self.num_row_choices*self.num_col_choices + 1) # +1 for done
        self.observation_space = Box(low=0, high=1, shape=(channels, 28, 28))#shape=(channels, 32, 32))
        

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, NEXT=True):
        if NEXT: # whether switch to next image
            self.curr_img, self.curr_label = next(iter(self.data_loader))
            while self.curr_label >= self.num_labels:
                self.curr_img, self.curr_label = next(iter(self.data_loader))
            self.curr_img = self.curr_img.squeeze(0)
            self.curr_label = self.curr_label.squeeze(0)

        # initialize position at center of image
        self.pos = [max(0, self.curr_img.shape[1]//2-self.window), max(0, self.curr_img.shape[2]//2-self.window)]
        self.state = -np.ones(
            (self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
        self.state[
            1:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
            self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window]
        self.num_steps = 0
        return self.state

    def step(self, action, curr_clf_softmax, prev_clf_softmax):
        # while len(self.curr_label.shape) < 2: #should be (n, 1), n=number of exampels
        #     self.curr_label = self.curr_label.unsqueeze_(dim=0)

        # if agent chooses to done with high confidence
        if action[0] == self.action_space.n and curr_clf_softmax.cpu().detach().numpy()[0][action[-1]] > 0.75:
            done = True
        else:
            done = False
            
            if action[0] <= self.num_row_choices * self.num_col_choices: # move
                self.pos[0] = min(self.curr_img.shape[1], self.window*(action[0] // self.num_col_choices))# row move
                self.pos[1] = min(self.curr_img.shape[2], self.window*(action[0] % self.num_col_choices))# col move

            else:
                print("Action out of bounds!")
                return
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1 # location channel
        self.state[1:,
            self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
                self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] # image channel
        self.num_steps += 1
        if not done: 
            done = self.num_steps >= self.max_steps

        
        if done and action[-1] == self.curr_label.item(): 
            reward = 1

        elif done and action[-1] != self.curr_label.item():
            reward = -10
        else: # reward for each step = cost_exploration + improvement(curr_pred_softmax, prev_pred_softmax)
            # curr_clf_loss = Hellinger_distance(curr_clf_softmax, self.curr_label.to(device))
            # prev_clf_loss = Hellinger_distance(prev_clf_softmax, self.curr_label.to(device))
            # pdb.set_trace()
            curr_clf_loss = F.nll_loss(torch.log(curr_clf_softmax), self.curr_label.unsqueeze(dim=0)).item()
            prev_clf_loss = F.nll_loss(torch.log(prev_clf_softmax), self.curr_label.unsqueeze(dim=0)).item()
            # print ('curr_clf_loss', curr_clf_loss)
            # print ('prev_clf_loss', prev_clf_loss)
            clf_improvement = -(curr_clf_loss - prev_clf_loss)
            reward = - 0.5*( 1 / self.max_steps) + 0.5*(clf_improvement / self.max_steps)
   
        return self.state, reward, done, {}

    def get_current_obs(self):
        return self.state

    def close(self):
        pass



if __name__ == '__main__':
    # transform = T.Compose([
    #                T.ToTensor(),
    #                T.Normalize((0.1307,), (0.3081,))
    #            ])
    # dataset = datasets.MNIST
    # channels = 2
    # train = True
    # loader = torch.utils.data.DataLoader(
    #     dataset('data', train=train, download=True,
    #         transform=transform),
    #     batch_size=60000, shuffle=True)
    # for imgs, labels in loader:
    #     break
    MAX_STEPS = 10
    env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=5, num_labels=10)
    
    # loader = torch.utils.data.DataLoader(
    #     Cityscapes(CITYSCAPE, train, transform=transform), batch_size=10000,
    #     shuffle=True)
    # for imgs, labels in loader:
    #     break
    # env = DetectionEnv(imgs, labels, max_steps=200)