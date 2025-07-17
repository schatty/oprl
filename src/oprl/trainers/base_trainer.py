from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch as t

from oprl.algos.protocols import AlgorithmProtocol
from oprl.environment import EnvProtocol
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.logging import LoggerProtocol, create_stdout_logger

from oprl.trainers.protocols import TrainerProtocol


logger = create_stdout_logger()


@dataclass
class BaseTrainer(TrainerProtocol):
    logger: LoggerProtocol
    env: EnvProtocol
    make_env_test: Callable[[int], EnvProtocol]
    replay_buffer: ReplayBufferProtocol
    algo: AlgorithmProtocol
    gamma: float = 0.99
    num_steps: int = int(1e6)
    start_steps: int = int(10e3)
    batch_size: int = 128
    eval_interval: int = int(2e3)
    num_eval_episodes: int = 10
    save_buffer_every: int = 0
    save_policy_every: int = int(100_000)
    estimate_q_every: int = 0
    stdout_log_every: int = int(1e5)
    device: str = "cpu"
    seed: int = 0

    def train(self) -> None:
        self.algo.check_created()
        self.replay_buffer.check_created()

        ep_step = 0
        state, _ = self.env.reset()
        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self.env.sample_action()
            else:
                action = self.algo.actor.explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.replay_buffer.add_transition(
                state, action, reward, terminated, episode_done=terminated or truncated
            )
            if terminated or truncated:
                next_state, _ = self.env.reset()
                ep_step = 0
            state = next_state

            if len(self.replay_buffer) < self.batch_size:
                continue

            (
                states,
                actions,
                rewards,
                dones,
                next_states
            ) = self.replay_buffer.sample(self.batch_size)
            self.algo.update(states, actions, rewards, dones, next_states)

            self._log_evaluation(env_step, rewards)
            self._save_policy(env_step)
            self._log_stdout(env_step, rewards)

    def _log_evaluation(self, env_step: int, rewards: t.Tensor) -> None:
        if env_step % self.eval_interval == 0:
            eval_metrics = self.evaluate()
            self.logger.log_scalar("trainer/ep_reward", eval_metrics["return"], env_step)
            self.logger.log_scalar("trainer/avg_reward", rewards.mean().item(), env_step)
            self.logger.log_scalar(
                "trainer/buffer_transitions", len(self.replay_buffer), env_step
            )
            self.logger.log_scalar(
                "trainer/buffer_episodes", self.replay_buffer.episodes_counter, env_step
            )
            self.logger.log_scalar(
                "trainer/buffer_last_ep_len",
                self.replay_buffer.last_episode_length,
                env_step,
            )

    def evaluate(self) -> dict[str, float]:
        returns = []
        for i_ep in range(self.num_eval_episodes):
            env_test = self.make_env_test(self.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0.0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.algo.actor.exploit(state)
                state, reward, terminated, truncated, _ = env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        return {
            "return": float(np.mean(returns))
        }

    def _save_policy(self, env_step: int) -> None:
        if self.save_policy_every > 0 and env_step % self.save_policy_every == 0:
            weights_path = self.logger.log_dir / "weights" / f"{env_step}.w"
            weights_path.parents[0].mkdir(exist_ok=True)
            t.save(
                self.algo.actor,
                weights_path
            )

    def _estimate_q(self, env_step: int) -> None:
        if self.estimate_q_every > 0 and env_step % self.estimate_q_every == 0:
            q_true = self.estimate_true_q()
            q_critic = self.estimate_critic_q()
            if q_true is not None:
                self.logger.log_scalar("trainer/Q-estimate", q_true, env_step)
                self.logger.log_scalar("trainer/Q-critic", q_critic, env_step)
                self.logger.log_scalar(
                    "trainer/Q_asb_diff", q_critic - q_true, env_step
                )

    def _log_stdout(self, env_step: int, rewards: t.Tensor) -> None:
        if env_step % self.stdout_log_every == 0:
            perc = int(env_step / self.num_steps * 100)
            logger.info(
                f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {rewards.mean():10.3f}"
            )

    def estimate_true_q(self, eval_episodes: int = 10) -> float:
        qs = []
        for i_eval in range(eval_episodes):
            env = self.make_env_test(self.seed * 100 + i_eval)
            state, _ = env.reset()

            q = 0
            s_i = 1
            while True:
                action = self.algo.actor.exploit(state)
                state, r, terminated, truncated, _ = env.step(action)
                q += r * self.gamma ** s_i
                s_i += 1
                if terminated or truncated:
                    break
            qs.append(q)

        return np.mean(qs, dtype=float)

    def estimate_critic_q(self, num_episodes: int = 10) -> float:
        qs = []
        for i_eval in range(num_episodes):
            env = self.make_env_test(self.seed * 100 + i_eval)
            state, _ = env.reset()
            action = self.algo.actor.exploit(state)

            state = t.tensor(state).unsqueeze(0).float().to(self.device)
            action = t.tensor(action).unsqueeze(0).float().to(self.device)

            q = self.algo.critic(state, action)
            # TODO: TQC is not supported by this logic, need to update
            if isinstance(q, tuple):
                q = q[0]
            q = q.item()
            qs.append(q)

        return np.mean(qs, dtype=float)
