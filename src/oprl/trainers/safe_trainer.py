from dataclasses import dataclass

import numpy as np

from oprl.trainers.base_trainer import BaseTrainer, TrainerProtocol


@dataclass
class SafeTrainer(TrainerProtocol):
    trainer: BaseTrainer

    def train(self):
        ep_step = 0
        state, _ = self.trainer.env.reset()
        total_cost = 0

        for env_step in range(self.trainer.num_steps + 1):
            ep_step += 1
            if env_step <= self.trainer.start_steps:
                action = self.trainer.env.sample_action()
            else:
                action = self.trainer.algo.explore(state)
            next_state, reward, terminated, truncated, info = self.trainer.env.step(action)
            total_cost += info["cost"]

            self.trainer.replay_buffer.add_transition(
                state, action, reward, terminated, episode_done=terminated or truncated
            )
            if terminated or truncated:
                next_state, _ = self.trainer.env.reset()
                ep_step = 0
            state = next_state

            if len(self.trainer.replay_buffer) < self.trainer.batch_size:
                continue
            batch = self.trainer.replay_buffer.sample(self.trainer.batch_size)
            self.trainer.algo.update(*batch)

            self._log_evaluation(env_step, batch)
            self.trainer._save_policy(env_step)
            self.trainer._log_stdout(env_step, batch)

        self.trainer.logger.log_scalar("trainer/total_cost", total_cost, self.trainer.num_steps)

    def _log_evaluation(self, env_step: int, batch) -> None:
        if env_step % self.trainer.eval_interval == 0:
            eval_metrics = self.evaluate()
            self.trainer.logger.log_scalar(
                "trainer/ep_reward", eval_metrics["return"], env_step
            )
            self.trainer.logger.log_scalar(
                "trainer/ep_cost", eval_metrics["cost"], env_step
            )

            self.trainer.logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
            self.trainer.logger.log_scalar(
                "trainer/buffer_transitions", len(self.trainer.replay_buffer), env_step
            )
            self.trainer.logger.log_scalar(
                "trainer/buffer_episodes", self.trainer.replay_buffer.episodes_counter, env_step
            )
            self.trainer.logger.log_scalar(
                "trainer/buffer_last_ep_len",
                self.trainer.replay_buffer.last_episode_length,
                env_step,
            )

    def evaluate(self) -> dict[str, float]:
        returns = []
        costs = []
        for i_ep in range(self.trainer.num_eval_episodes):
            env_test = self.trainer.make_env_test(seed=self.trainer.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0
            episode_cost = 0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.trainer.algo.exploit(state)
                state, reward, terminated, truncated, info = env_test.step(action)
                episode_return += reward
                episode_cost += info["cost"]

            returns.append(episode_return)
            costs.append(episode_cost)

        return {
            "return": float(np.mean(returns)),
            "cost": float(np.mean(costs)),
        }

