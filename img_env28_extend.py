import numpy as np

import torch
from gym.spaces import Discrete, Box
import torchvision.transforms as T
from torchvision import datasets
import matplotlib.pyplot as plt


CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']


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
        # Extend action space with the 5th choice: if action = 4, the agent is ready to 
        # get its prediction checked
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=1, shape=(channels, 28, 28))#shape=(channels, 32, 32))
        self.channels = channels
        self.data_loader = get_data_loader(dataset, train=train)
        self.window = window
        self.max_steps = max_steps
        self.num_labels = num_labels

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
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

    def step(self, action):
        done = False
        # Go up
        if action[0] == 0:
            self.pos[0] = max(0, self.pos[0] - self.window)

        # Go down
        elif action[0] == 1:
            self.pos[0] = min(self.curr_img.shape[1] - self.window,
                              self.pos[0] + self.window)
        # Go left
        elif action[0] == 2:
            self.pos[1] = max(0, self.pos[1] - self.window)

        # Go right
        elif action[0] == 3:
            self.pos[1] = min(self.curr_img.shape[2] - self.window,
                              self.pos[1] + self.window)

        # Ready to predict, go nowhere
        elif action[0] == 4:
            done = True
        else:
            print("Action out of bounds!")
            return
        self.state[0, :, :] = np.zeros(
            (1, self.curr_img.shape[1], self.curr_img.shape[2]))
        self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
        self.state[1:,
            self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
                self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window]
        self.num_steps += 1
        if not done: 
            done = self.num_steps >= self.max_steps
        
        if done and action[1] == self.curr_label.item():
            reward = 1
        elif done and action[1] != self.curr_label.item():
            reward = -1
        else:
            reward = - 1 / self.max_steps
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
    # env = ImgEnv_extend(imgs, labels, max_steps=200, channels=channels, window=5)
    MAX_STEPS = 10
    env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=14, num_labels=10)
    observation = env.reset()
    for t in range(MAX_STEPS):
        move = np.random.choice((range(5)))
        clf = np.random.choice((range(10)))
        actionS = [move, clf]
        observation, reward, done, _ = env.step(actionS)
        print ('t = {t}, actionS = {actionS}, done = {done}, reward = {reward}'.format(**locals()))
        plt.imshow(observation[1,:,:])
        plt.show()
        if done:
            break

    # loader = torch.utils.data.DataLoader(
    #     Cityscapes(CITYSCAPE, train, transform=transform), batch_size=10000,
    #     shuffle=True)
    # for imgs, labels in loader:
    #     break
    # env = DetectionEnv(imgs, labels, max_steps=200)