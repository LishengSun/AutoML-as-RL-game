import os
import sys

envs = ['cifar10'] # ['mnist', 'cifar100', 'cityscapes'] #
algos = ['ppo', 'a2c', 'acktr'] #
windows = [5] # [2, 7]
max_steps = 20
max_steps_ls = [1] # [10, 20, 40]
window = 32
for env in envs:
    if env == 'cityscapes':
        num_processes = 2
    else:
        num_processes = 8
    for algo in algos:
        for window in windows:
            for seed in range(5):
                cmd = 'sbatch ./slurm.sh' + \
                    ' --seed {seed}' + \
                    ' --env-name {env}' + \
                    ' --algo {algo}' + \
                    ' --num-processes {num_processes}' + \
                    ' --num-stack 1' + \
                    ' --log-interval 100' + \
                    ' --out-dir /checkpoint/amyzhang/natural_rl/models'
                cmd += ' --window {window} --max-steps {max_steps}'
                # cmd += ' --resnet'
                # if env == 'cityscapes':
                #     cmd += ' --no-cuda'
                if algo == 'ppo':
                    cmd += ' --use-gae --lr 2.5e-4 --clip-param 0.1'
                    cmd += ' --value-loss-coef 1'
                    cmd += ' --num-steps 128 --num-mini-batch 4 --vis-interval 1'
                    cmd += ' --log-interval 10'
                elif algo == 'acktr':
                    cmd += ' --num-steps 20'
                    cmd += ' --log-interval 100'

                os_cmd = cmd.format(**locals())
                os.system(os_cmd)
                if seed == 1: print(os_cmd)
