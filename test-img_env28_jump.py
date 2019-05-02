from img_env28_jump import ImgEnv
import matplotlib.pyplot as plt
import numpy as np
import torch

import model_extend
import pdb


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_STEPS = 16
    env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=8, num_labels=10)
    myClassifier = model_extend.myClassifier_with_CNNpretrained()
    observation = env.reset()
    fig, ax = plt.subplots()
    total_reward = 0
    GAMMA = 0.99
    prev_clf_softmax = [1/10]*10
    prev_clf_softmax = torch.Tensor(prev_clf_softmax).unsqueeze(dim=0)
    for t in range(MAX_STEPS):
        inputs = torch.from_numpy(observation).float().resize_(1, observation.shape[0], observation.shape[1], observation.shape[2]).to(device)    
        log_curr_clf_softmax, curr_clf_softmax = myClassifier(inputs[:,1:2,:,:])
        pred_label = [curr_clf_softmax.data.max(1, keepdim=True)[1].item()]
        done = [0]
        move = [t] + [0]
        actionS = move + pred_label
        observation, reward, done, _ = env.step(actionS, curr_clf_softmax, prev_clf_softmax)
        prev_clf_softmax = curr_clf_softmax
        row_move = actionS[0] // 4
        col_move = actionS[0] % 4
        
        clf_prob = curr_clf_softmax[0][actionS[-1]]
        total_reward += GAMMA * reward
        rew = round(reward, 4)
        tot_rew = round(total_reward, 2)
        ax.imshow(observation[1,:,:]) # show image
        ax.set_title('t = {t}, a = {actionS}, r = {rew}, R_tot = {tot_rew}'.format(**locals()))
        # plt.show()
        plt.pause(3)
        if done:
            break
