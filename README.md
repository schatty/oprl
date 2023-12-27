# D4PG-pytorch

PyTorch implementation of Distributed Distributional Deterministic Policy Gradients (https://arxiv.org/abs/1804.08617).
<p align="center">
<img width="400" alt="d4pg_arch" src="https://user-images.githubusercontent.com/23639048/137602300-f2763ef1-2b67-4f76-aa8b-232afaa08a23.png">
</p>


# About
The project is under an active renovation, for the old code with `D4PG` algorithm working with multiprocessing queues and `mujoco_py` please refer to the branch `d4pg_legacy`.

### Roadmap 🏗
- [x] Switching to `mujoco 3.1.1`
- [x] Replacing multiprocessing queues with RabbitMQ for distributed RL
- [x] Baselines with DDPG, TQC for `dm_control` for 1M step
- [ ] Baselines with Distributed DDPG for `dm_control`
- [ ] Bringing back D4PG logic on top of TQC
- [ ] Tests
- [ ] New Algos

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
```
python src/oprl/configs/d3pg.py --env walker-walk
```

## Results

Results for single process DDPG and TQC:
![ddpg_tqc_eval](https://github.com/schatty/d4pg-pytorch/assets/23639048/f2c32f62-63b4-4a66-a636-4ce0ea1522f6)

## References
* Continuous control with deep reinforcement learning, [https://arxiv.org/abs/1509.02971]
* Distributed Distributional Deterministic Policy Gradients [https://arxiv.org/abs/1804.08617]
