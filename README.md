# D4PG-pytorch

PyTorch implementation of Distributed Distributional Deterministic Policy Gradients (https://arxiv.org/abs/1804.08617).
<img width="865" alt="arch" src="https://user-images.githubusercontent.com/23639048/62874932-bedde500-bd2a-11e9-82e3-6b4899b4e6d2.png">
Supported environments
* Pendulum-v0
* LunarLanderContinous-v2
* BipedalWalker-v2

# Usage
Run `train.py` to run experiment specified in `config.yaml`.

# Tests
In progress, for now tests can be used for reproducing results.

## Demo
![demo](https://user-images.githubusercontent.com/23639048/62875572-eda88b00-bd2b-11e9-95f7-b47d9522df33.gif)

Detailed results of training can be found at
* Pendulum-v0
  * [d3pg](https://schatty.github.io/d4pg/pendulum_d3pg/), [d3pg prioritized](https://schatty.github.io/d4pg/pendulum_d3pg_prior/)
  * [d4pg](https://schatty.github.io/d4pg/pendulum_d4pg/), [d4pg prioritized](https://schatty.github.io/d4pg/pendulum_d4pg_prior/)
* LunarLanderContinuous-v2
  * [d3pg](https://schatty.github.io/d4pg/lunarlander_d3pg/), [d3pg prioritized](https://schatty.github.io/d4pg/lunarlander_d3pg_prior/)
  * [d4pg](https://schatty.github.io/d4pg/lunarlander_d4pg/), [d4pg prioritized](https://schatty.github.io/d4pg/lunarlander_d4pg_prior/)
* BipedalWalker-v2
  * [d3pg](https://schatty.github.io/d4pg/bipedal_d3pg/), [d3pg prioritized](https://schatty.github.io/d4pg/bipedal_d3pg_prior/)
  * [d4pg](https://schatty.github.io/d4pg/bipedal_d4pg/), [d4pg prioritized](https://schatty.github.io/d4pg/bipedal_d4pg_prior/)

## Acknowledgements
The project partly based on the [Mark Sinton](https://github.com/msinto93) TensorFlow implementation, which helped greatly in difficult parts.
