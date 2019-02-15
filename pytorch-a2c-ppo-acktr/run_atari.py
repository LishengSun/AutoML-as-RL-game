import os
import sys

envs = [
    'AlienNoFrameskip-v4',
    'AirRaidNoFrameskip-v4',
    'AmidarNoFrameskip-v4',
    'AtlantisNoFrameskip-v4',
    'AssaultNoFrameskip-v4',
    'AsteroidsNoFrameskip-v4',
    'BeamRiderNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    'CarnivalNoFrameskip-v4',
    'CentipedeNoFrameskip-v4',
    'DemonAttackNoFrameskip-v4',
    'GravitarNoFrameskip-v4',
    'JourneyEscapeNoFrameskip-v4',
    'PhoenixNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'StarGunnerNoFrameskip-v4'
    ]
algos = ['ppo', 'acktr', 'a2c']
for env in envs:
    for algo in algos:
        for seed in range(5):
            cmd = 'sbatch ./slurm.sh' + \
                ' --seed {seed}' + \
                ' --env-name {env}' + \
                ' --algo {algo}' + \
                ' --out-dir /checkpoint/amyzhang/natural_rl/models'
            cmd += ' --resnet'
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
