from img_env28_jump import ImgEnv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    MAX_STEPS = 10
    env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=5, num_labels=10)
    observation = env.reset()
    for t in range(MAX_STEPS):
        # move = np.random.choice((range(785)))
        move = [np.random.choice((range(28))), np.random.choice((range(28))), np.random.choice((range(2)))]
        clf = [np.random.choice((range(10)))]
        # actionS = [move, clf]
        actionS = move + clf
        observation, reward, done, _ = env.step(actionS)
        # row_move = actionS[0] // 28
        # col_move = actionS[0] % 28
        row_move = actionS[0]
        col_move = actionS[1]
        print ('t = {t}, actionS = {actionS}, row = {row_move}, col = {col_move}, done = {done}, reward = {reward}'.format(**locals()))
        plt.imshow(observation[1,:,:]) # show image
        plt.show()
        if done:
            break
