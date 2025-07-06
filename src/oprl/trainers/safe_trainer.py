from typing import Any, Callable

import numpy as np

from oprl.env import BaseEnv
from oprl.trainers.base_trainer import BaseTrainer
from oprl.logging import LoggerProtocol


class SafeTrainer:
    def __init__(
        self,
        trainer: BaseTrainer
    ):
        self.trainer = trainer

    def train(self):
        ep_step = 0
        state, _ = self.trainer._env.reset()
        total_cost = 0

        for env_step in range(self.trainer.num_steps + 1):
            ep_step += 1
            if env_step <= self.trainer.start_steps:
                action = self.trainer._env.sample_action()
            else:
                action = self.trainer._algo.explore(state)
            next_state, reward, terminated, truncated, info = self.trainer._env.step(action)
            total_cost += info["cost"]

            self.trainer.replay_buffer.add_transition(
                state, action, reward, terminated, episode_done=terminated or truncated
            )
            if terminated or truncated:
                next_state, _ = self.trainer._env.reset()
                ep_step = 0
            state = next_state

            if len(self.trainer.replay_buffer) < self.trainer.batch_size:
                continue
            batch = self.trainer.replay_buffer.sample(self.trainer.batch_size)
            self.trainer._algo.update(*batch)

            self._eval_routine(env_step, batch)
            self.trainer._save_policy(env_step)
            self.trainer._save_buffer(env_step)
            self.trainer._log_stdout(env_step, batch)

        self.trainer._logger.log_scalar("trainer/total_cost", total_cost, self.trainer.num_steps)

    def _eval_routine(self, env_step: int, batch):
        if env_step % self.trainer.eval_interval == 0:
            self._log_evaluation(env_step)

            self.trainer._logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
            self.trainer._logger.log_scalar(
                "trainer/buffer_transitions", len(self.trainer.replay_buffer), env_step
            )
            self.trainer._logger.log_scalar(
                "trainer/buffer_episodes", self.trainer.replay_buffer.cur_episodes, env_step
            )
            self.trainer._logger.log_scalar(
                "trainer/buffer_last_ep_len",
                self.trainer.replay_buffer.last_episode_length,
                env_step,
            )

    def _log_evaluation(self, env_step: int):
        returns = []
        costs = []
        for i_ep in range(self.trainer.num_eval_episodes):
            env_test = self.trainer._make_env_test(seed=self.trainer.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0
            episode_cost = 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.trainer._algo.exploit(state)
                state, reward, terminated, truncated, info = env_test.step(action)
                episode_return += reward
                episode_cost += info["cost"]

            returns.append(episode_return)
            costs.append(episode_cost)

        self.trainer._logger.log_scalar(
            "trainer/ep_reward", np.mean(returns, dtype=float), env_step
        )
        self.trainer._logger.log_scalar(
            "trainer/ep_cost", np.mean(costs, dtype=float), env_step
        )
