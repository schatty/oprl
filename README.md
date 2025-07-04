<img align="left" width="70" alt="oprl_logo" src="https://github.com/schatty/oprl/assets/23639048/c7ea0fee-3472-4d9c-86f3-9ab01f02222d">

# OPRL

A Modular Library for Off-Policy Reinforcement Learning with a focus on SafeRL and distributed computing. Benchmarking resutls are available at associated homepage: [Homepage](https://schatty.github.io/oprl/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/schatty/oprl/branch/master/graph/badge.svg)](https://codecov.io/gh/schatty/oprl)



# Disclaimer 
The project is under an active renovation, for the old code with D4PG algorithm working with multiprocessing queues and `mujoco_py` please refer to the branch `d4pg_legacy`.

### Roadmap 🏗
- [x] Switching to `mujoco 3.1.1`
- [x] Replacing multiprocessing queues with RabbitMQ for distributed RL
- [x] Baselines with DDPG, TQC for `dm_control` for 1M step
- [x] Tests
- [x] Support for SafetyGymnasium
- [ ] Style and readability improvements
- [ ] Baselines with Distributed algorithms for `dm_control`
- [ ] D4PG logic on top of TQC

# Installation

```
pip install -r requirements.txt
cd src && pip install -e .
```

For working with [SafetyGymnasium](https://github.com/PKU-Alignment/safety-gymnasium) install it manually
```
git clone https://github.com/PKU-Alignment/safety-gymnasium
cd safety-gymnasium && pip install -e .
```

# Usage

To run DDPG in a single process
```
python src/oprl/configs/ddpg.py --env walker-walk
```

To run distributed DDPG

Run RabbitMQ
```
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management
```

Run training
```
python src/oprl/configs/d3pg.py --env walker-walk
```

## Tests

```
cd src && pip install -e .
cd .. && pip install -r tests/functional/requirements.txt
python -m pytest tests
```

## Results

Results for single process DDPG and TQC:
![ddpg_tqc_eval](https://github.com/schatty/d4pg-pytorch/assets/23639048/f2c32f62-63b4-4a66-a636-4ce0ea1522f6)

## Acknowledgements
* DDPG and TD3 code is based on the official TD3 implementation: [sfujim/TD3](https://github.com/sfujim/TD3)
* TQC code is based on the official TQC implementation: [SamsungLabs/tqc](https://github.com/SamsungLabs/tqc)
* SafetyGymnasium: [PKU-Alignment/safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
