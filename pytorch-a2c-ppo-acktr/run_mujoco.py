import os
import sys

envs = [
    'Swimmer-v2',
    'Ant-v2',
    'Hopper-v2',
    'HalfCheetah-v2'
    ]
algos = ['acktr'] # 'ppo', 'a2c']
for env in envs:
    for algo in algos:
        for seed in range(5):
            cmd = 'sbatch ./slurm.sh' + \
                ' --seed {seed}' + \
                ' --env-name {env}' + \
                ' --algo {algo}' + \
                ' --num-stack 4' + \
                ' --out-dir /checkpoint/amyzhang/natural_rl/models'
                # ' --resnet'
            if algo == 'ppo':
                cmd += ' --use-gae --lr 2.5e-4 --clip-param 0.1'
                cmd += ' --value-loss-coef 1 --num-processes 8'
                cmd += ' --num-steps 128 --num-mini-batch 4 --vis-interval 1'
                cmd += ' --log-interval 10'
            elif algo == 'acktr':
                cmd += ' --num-processes 32 --num-steps 20'
            os_cmd = cmd.format(**locals())
            os.system(os_cmd)
            if seed == 1: print(os_cmd)

            cmd += ' --nat'
            cmd += ' --tag driving'
            os_cmd = cmd.format(**locals())
            os.system(os_cmd)
            if seed == 1: print(os_cmd)
