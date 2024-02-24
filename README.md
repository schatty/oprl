<img align="left" width="75" alt="oprl_logo" src="https://github.com/schatty/oprl/assets/23639048/c7ea0fee-3472-4d9c-86f3-9ab01f02222d">

# OPRL

A Modular Library for Off-Policy Reinforcement Learning with a focus on SafeRL and distributed computing.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Disclaimer 
The project is under an active renovation, for the old code with D4PG algorithm working with multiprocessing queues and `mujoco_py` please refer to the branch `d4pg_legacy`.

### Roadmap 🏗
- [x] Switching to `mujoco 3.1.1`
- [x] Replacing multiprocessing queues with RabbitMQ for distributed RL
- [x] Baselines with DDPG, TQC for `dm_control` for 1M step
- [x] Tests
- [ ] Support for SafetyGymnasium
- [ ] Baselines with Distributed algorithms for `dm_control`
- [ ] D4PG logic on top of TQC

# Installation

```
pip install -r requirements.txt
cd src && pip install -e .
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

## References
* Continuous control with deep reinforcement learning, [https://arxiv.org/abs/1509.02971]
* Distributed Distributional Deterministic Policy Gradients [https://arxiv.org/abs/1804.08617]
