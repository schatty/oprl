# D4PG-pytorch

PyTorch implementation of Distributed Distributional Deterministic Policy Gradients (https://arxiv.org/abs/1804.08617).
<p align="center">
<img width="400" alt="d4pg_arch" src="https://user-images.githubusercontent.com/23639048/137602300-f2763ef1-2b67-4f76-aa8b-232afaa08a23.png">
</p>


# About
The project is under an active renovation 🏗, for the old code working with multiprocessing queues and `mujoco_py` please refer to the branch `d4pg_legacy`.

### Roadmap
- [x] Switching to `mujoco 2.3.7` 
- [ ] Baselines with DDPG for `dm_control` for 1M step
- [ ] Baselines with Distributed DDPG for `dm_control`
- [ ] Bringing back D4PG logic
- [ ] Tests
- [ ] New Algos

# Usage

To run DDPG in a single process
```
python src/train.py --config src/configs/ddpg_walker.py --env walker-walk --n_seeds_processes 1
```

To run distributed DDPG
```
python src/oprl/configs/d3pg.py --env walker-walk
```

## Results

Preliminary results of D3PG for `walker` domain
<p align="center">
<img width="1000" alt="prelim_results_d3pg" src="https://github.com/schatty/d4pg-pytorch/assets/23639048/fe3057c7-4792-41fe-98f6-abc8e5ccb710">
</p>

## References
* Continuous control with deep reinforcement learning, [https://arxiv.org/abs/1509.02971]
* Distributed Distributional Deterministic Policy Gradients [https://arxiv.org/abs/1804.08617]
