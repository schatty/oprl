<p align="center">
    <img src="https://github.com/user-attachments/assets/0c98353f-3de6-46f6-b40c-db1d69672b12" alt="Description" width="150">
</p>

A Modular Library for Off-Policy Reinforcement Learning with a focus on SafeRL and distributed computing. The code supports `SafetyGymnasium` environment set for giving a starting point developing SafeRL solutions. Distributed setting is implemented via `pika` library and will be improved in the near future.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/schatty/oprl/branch/master/graph/badge.svg)](https://codecov.io/gh/schatty/oprl)

### Roadmap 🏗
- [x] Support for SafetyGymnasium
- [x] Style and readability improvements
- [ ] REDQ, DrQ Algorithms support
- [ ] Distributed Training Improvements

## In a Snapshot

Environments Support

| DMControl Suite | SafetyGymnasium | Gymnasium |
| -------- | -------- | -------- |

Algorithms

| DDPG | TD3 | SAC | TQC |
| --- | --- | --- | --- |

## Installation

The project supports [uv](https://docs.astral.sh/uv/) for package managment and [ruff](https://github.com/astral-sh/ruff) for formatting checks. To install it via uv in virutalenv: 

```
uv venv
source .venv/bin/activate
uv sync
```

### Installing SafetyGymnasium

For working with [SafetyGymnasium](https://github.com/PKU-Alignment/safety-gymnasium) install it manually
```
git clone https://github.com/PKU-Alignment/safety-gymnasium
cd safety-gymnasium && uv pip install -e .
```

## Tests

To run tests locally:

```
uv pip install pytest
uv run pytest tests/functional
```

## RL Training

All training is set via python config files located in `configs` folder. To make your own configuration, change the code there or create a similar one. During training, all the code is copied to logs folder to ensure full experimental reproducibility.

### Single Agent

To run DDPG in a single process
```
python configs/ddpg.py --env walker-walk
```

### Distributed

Run RabbitMQ
```
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management
```

Run training
```
python configs/distrib_ddpg.py --env walker-walk
```

## Results

Results for single process DDPG and TQC:
![ddpg_tqc_eval](https://github.com/schatty/d4pg-pytorch/assets/23639048/f2c32f62-63b4-4a66-a636-4ce0ea1522f6)

## Cite

__OPRL__
```
@inproceedings{
  kuznetsov2024safer,
  title={Safer Reinforcement Learning by Going Off-policy: a Benchmark},
  author={Igor Kuznetsov},
  booktitle={ICML 2024 Next Generation of AI Safety Workshop},
  year={2024},
  url={https://openreview.net/forum?id=pAmTC9EdGq}
}
```

__SafetyGymnasium__
```
@inproceedings{ji2023safety,
  title={Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark},
  author={Jiaming Ji and Borong Zhang and Jiayi Zhou and Xuehai Pan and Weidong Huang and Ruiyang Sun and Yiran Geng and Yifan Zhong and Josef Dai and Yaodong Yang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=WZmlxIuIGR}
}
```
