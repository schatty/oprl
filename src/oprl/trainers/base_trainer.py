from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import torch

from oprl.algos import OffPolicyAlgorithm
from oprl.environment import EnvProtocol
from oprl.buffers.episodic_buffer import ReplayBufferProtocol
from oprl.logging import LoggerProtocol


class TrainerProtocol(Protocol):
    def train(self) -> None: ...

    def evaluate(self) -> dict[str, float]: ...


@dataclass
class BaseTrainer(TrainerProtocol):
    env: EnvProtocol
    make_env_test: Callable[[int], EnvProtocol]
    replay_buffer: ReplayBufferProtocol
    algo: OffPolicyAlgorithm | None = None
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
    logger: LoggerProtocol | None = None

    def train(self) -> None:
        ep_step = 0
        state, _ = self.env.reset()

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self.env.sample_action()
            else:
                action = self.algo.explore(state)
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

            batch = self.replay_buffer.sample(self.batch_size)
            self.algo.update(*batch)

            self._log_evaluation(env_step, batch)
            self._save_policy(env_step)
            self._log_stdout(env_step, batch)

    def _log_evaluation(self, env_step: int, batch):
        if env_step % self.eval_interval == 0:
            eval_metrics = self.evaluate()
            self.logger.log_scalar("trainer/ep_reward", eval_metrics["return"], env_step)

            self.logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
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
            env_test = self.make_env_test(seed=self.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0.0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        return {
            "return": float(np.mean(returns))
        }

    def _save_policy(self, env_step: int):
        if self.save_policy_every > 0 and env_step % self.save_policy_every == 0:
            self.logger.save_weights(self.algo.actor, env_step)

    def _estimate_q(self, env_step: int):
        if self.estimate_q_every > 0 and env_step % self.estimate_q_every == 0:
            q_true = self.estimate_true_q()
            q_critic = self.estimate_critic_q()
            if q_true is not None:
                self.logger.log_scalar("trainer/Q-estimate", q_true, env_step)
                self.logger.log_scalar("trainer/Q-critic", q_critic, env_step)
                self.logger.log_scalar(
                    "trainer/Q_asb_diff", q_critic - q_true, env_step
                )

    def _log_stdout(self, env_step: int, batch):
        if env_step % self.stdout_log_every == 0:
            perc = int(env_step / self.num_steps * 100)
            print(
                f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {batch[2].mean():10.3f}"
            )

    def estimate_true_q(self, eval_episodes: int = 10) -> float | None:
        try:
            qs = []
            for i_eval in range(eval_episodes):
                env = self.make_env_test(seed=self.seed * 100 + i_eval)
                print("Before reset etimate q")
                state, _ = env.reset()

                q = 0
                s_i = 1
                while True:
                    action = self.algo.exploit(state)
                    state, r, terminated, truncated, _ = env.step(action)
                    q += r * self.gamma ** s_i
                    s_i += 1
                    if terminated or truncated:
                        break

                qs.append(q)

            return np.mean(qs, dtype=float)
        except Exception as e:
            print(f"Failed to estimate Q-value: {e}")
            return None

    def estimate_critic_q(self, num_episodes: int = 10) -> float:
        qs = []
        for i_eval in range(num_episodes):
            env = self.make_env_test(seed=self.seed * 100 + i_eval)

            state, _ = env.reset()
            action = self.algo.exploit(state)

            state = torch.tensor(state).unsqueeze(0).float().to(self.device)
            action = torch.tensor(action).unsqueeze(0).float().to(self.device)

            q = self.algo.critic(state, action)
            # TODO: TQC is not supported by this logic, need to update
            if isinstance(q, tuple):
                q = q[0]
            q = q.item()
            qs.append(q)

        return np.mean(qs, dtype=float)


def run_training(makealgo, makeenv, make_replay_buffer, makelogger, config: dict[str, Any], seed: int):
    env = makeenv(seed=seed)
    logger = makelogger(seed)

    trainer = BaseTrainer(
        env=env,
        makeenv_test=makeenv,
        algo=makealgo(logger, seed),
        replay_buffer=make_replay_buffer(),
        num_steps=config["num_steps"],
        eval_interval=config["eval_every"],
        device=config["device"],
        save_buffer_every=config["save_buffer"],
        estimate_q_every=config["estimate_q_every"],
        stdout_log_every=config["log_every"],
        seed=seed,
        logger=logger,
    )

    trainer.train()
