# D4PG-pytorch

PyTorch implementation of Distributed Distributional Deterministic Policy Gradients (https://arxiv.org/abs/1804.08617).
<img width="865" alt="arch" src="https://user-images.githubusercontent.com/23639048/62874932-bedde500-bd2a-11e9-82e3-6b4899b4e6d2.png">

Supported environments
* Pendulum-v0
* LunarLanderContinous-v2
* BipedalWalker-v2

# About
D4PG and D3PG implementations with following features
* learner, sampler and agents run in separate processes
* exploiter agent(s) exists which acts without noise in actions on target network
* GPU is hold only by exploiters, all other exploration processes are run on CPU

# Usage
Run `train.py --config configs/pendulum_d4pg.yml`

# Tests
`python -m unittest discover`

## Results
<img width="1014" alt="curves" src="https://user-images.githubusercontent.com/23639048/68063948-7b41b880-fd26-11e9-839f-f042dc54de0e.png">

## Reproduce
All results were obtained with configs in `configs` directory

## References

* DDPG [https://arxiv.org/abs/1509.02971]
* Distributional Perspective on RL [https://arxiv.org/abs/1804.08617]
* D4PG [https://arxiv.org/abs/1804.08617]
