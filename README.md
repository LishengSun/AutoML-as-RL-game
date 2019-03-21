# Using pretrained CNN and extra action 'done' to let the agent decide when to switch images

## The architecture:
![alt text](https://raw.githubusercontent.com/LishengSun/AutoML-as-RL-game/pretrainedCNN_with_extended_actions/figs/architecture_pretrainedCNN.png)

## Extended action space:
We added binary choices 'done' as part of the actions that allows the agent to decide when to terminate an episode, get its classification checked and switch to the next. The extended action space = {Up, Down, Left, Right, done}. done is 0 by default, the agent set it to 1 when it wants to terminate.

## Results:

![alt text](https://raw.githubusercontent.com/LishengSun/AutoML-as-RL-game/pretrainedCNN_with_extended_actions/figs/pretrainedCNN_extend_2labs_3runs_1000epis_10steps_8ws_rw-10.png)


