import numpy as np
import torch

from oprl.trainers.base_trainer import BaseTrainer
from oprl.trainers.buffers.episodic_buffer import EpisodicReplayBuffer


class SafeTrainer(BaseTrainer):
    def __init__(
        self,
        state_shape=None,
        action_shape=None,
        env=None,
        make_env_test=None,
        algo=None,
        buffer_size=int(1e6),
        gamma=0.99,
        device=None,
        num_steps=int(1e6),
        start_steps=int(10e3),
        batch_size=128,
        eval_interval=int(2e3),
        num_eval_episodes=10,
        save_buffer_every=0,
        visualise_every=0,
        estimate_q_every=0,
        stdout_log_every=int(1e5),
        seed=0,
        logger=None,
    ):
        """
        Args:
            state_shape: Shape of the state.
            action_shape: Shape of the action.
            env: Enviornment object.
            make_env_test: Environment object for evaluation.
            algo: Codename for the algo (SAC).
            buffer_size: Buffer size in transitions.
            gamma: Discount factor.
            device: Name of the device.
            num_step: Number of env steps to train.
            start_steps: Number of environment steps not to perform training at the beginning.
            batch_size: Batch-size.
            eval_interval: Number of env step after which perform evaluation.
            save_buffer_every: Number of env steps after which save replay buffer.
            visualise_every: Number of env steps after which perform vizualisation.
            stdout_log_every: Number of evn steps after which log info to stdout.
            seed: Random seed.
        """
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            env=env,
            make_env_test=make_env_test,
            algo=algo,
            buffer_size=buffer_size,
            gamma=gamma,
            device=device,
            num_steps=num_steps,
            start_steps=start_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_episodes,
            save_buffer_every=save_buffer_every,
            visualise_every=visualise_every,
            estimate_q_every=estimate_q_every,
            stdout_log_every=stdout_log_every,
            seed=seed,
            logger=logger,
        )

    def train(self):
        ep_step = 0
        state, _ = self.env.reset()
        total_cost = 0

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self.env.sample_action()
            else:
                action = self.algo.explore(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_cost += info["cost"]

            self.buffer.append(
                state, action, reward, terminated, episode_done=terminated or truncated
            )
            if terminated or truncated:
                next_state, _ = self.env.reset()
                ep_step = 0
            state = next_state

            if len(self.buffer) < self.batch_size:
                continue
            batch = self.buffer.sample(self.batch_size)
            self.algo.update(batch)

            self._eval_routine(env_step, batch)
            self._visualize(env_step)
            self._save_buffer(env_step)
            self._log_stdout(env_step, batch)

        self.logger.log_scalar("trainer/total_cost", total_cost, self.num_steps)

    def _log_evaluation(self, env_step: int):
        returns = []
        costs = []
        for i_ep in range(self.num_eval_episodes):
            env_test = self.make_env_test(seed=self.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0
            episode_cost = 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, info = env_test.step(action)
                episode_return += reward
                episode_cost += info["cost"]

            returns.append(episode_return)
            costs.append(episode_cost)

        self.logger.log_scalar("trainer/ep_reward", np.mean(returns), env_step)
        self.logger.log_scalar("trainer/ep_cost", np.mean(costs), env_step)
