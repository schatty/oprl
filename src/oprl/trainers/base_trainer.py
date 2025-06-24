from typing import Any, Callable

import numpy as np
import torch

from oprl.env import BaseEnv
from oprl.trainers.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.utils.logger import Logger, StdLogger


class BaseTrainer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env: BaseEnv,
        make_env_test: Callable[[int], BaseEnv],
        algo: Any | None = None,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        num_steps: int = int(1e6),
        start_steps: int = int(10e3),
        batch_size: int = 128,
        eval_interval: int = int(2e3),
        num_eval_episodes: int = 10,
        save_buffer_every: int = 0,
        visualise_every: int = 0,
        estimate_q_every: int = 0,
        stdout_log_every: int = int(1e5),
        device: str = "cpu",
        seed: int = 0,
        logger: Logger = StdLogger(),
    ):
        """
        Args:
            state_dim: Dimension of the observation.
            action_dim: Dimension of the action.
            env: Enviornment object.
            make_env_test: Environment object for evaluation.
            algo: Codename for the algo (SAC).
            buffer_size: Buffer size in transitions.
            gamma: Discount factor.
            num_step: Number of env steps to train.
            start_steps: Number of environment steps not to perform training at the beginning.
            batch_size: Batch-size.
            eval_interval: Number of env step after which perform evaluation.
            save_buffer_every: Number of env steps after which save replay buffer.
            visualise_every: Number of env steps after which perform vizualisation.
            device: Name of the device.
            stdout_log_every: Number of evn steps after which log info to stdout.
            seed: Random seed.
            logger: Logger instance.
        """
        self._env = env
        self._make_env_test = make_env_test
        self._algo = algo
        self._gamma = gamma
        self._device = device
        self._save_buffer_every = save_buffer_every
        self._visualize_every = visualise_every
        self._estimate_q_every = estimate_q_every
        self._stdout_log_every = stdout_log_every
        self._logger = logger
        self.seed = seed

        self.buffer = EpisodicReplayBuffer(
            buffer_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            gamma=gamma,
        )

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        ep_step = 0
        state, _ = self._env.reset()

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self._env.sample_action()
            else:
                action = self._algo.explore(state)
            next_state, reward, terminated, truncated, _ = self._env.step(action)

            self.buffer.append(
                state, action, reward, terminated, episode_done=terminated or truncated
            )
            if terminated or truncated:
                next_state, _ = self._env.reset()
                ep_step = 0
            state = next_state

            if len(self.buffer) < self.batch_size:
                continue

            batch = self.buffer.sample(self.batch_size)
            self._algo.update(*batch)

            self._eval_routine(env_step, batch)
            self._visualize(env_step)
            self._save_buffer(env_step)
            self._log_stdout(env_step, batch)

    def _eval_routine(self, env_step: int, batch):
        if env_step % self.eval_interval == 0:
            self._log_evaluation(env_step)

            self._logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
            self._logger.log_scalar(
                "trainer/buffer_transitions", len(self.buffer), env_step
            )
            self._logger.log_scalar(
                "trainer/buffer_episodes", self.buffer.num_episodes, env_step
            )
            self._logger.log_scalar(
                "trainer/buffer_last_ep_len",
                self.buffer.get_last_ep_len(),
                env_step,
            )

    def _log_evaluation(self, env_step: int):
        returns = []
        for i_ep in range(self.num_eval_episodes):
            env_test = self._make_env_test(seed=self.seed + i_ep)
            state, _ = env_test.reset()

            episode_return = 0.0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self._algo.exploit(state)
                state, reward, terminated, truncated, _ = env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self._logger.log_scalar("trainer/ep_reward", mean_return, env_step)

    def _visualize(self, env_step: int):
        if self._visualize_every > 0 and env_step % self._visualize_every == 0:
            imgs = self.visualise_policy()  # [T, W, H, C]
            if imgs is not None:
                self._logger.log_video("eval_policy", imgs, env_step)

    def _save_buffer(self, env_step: int):
        if self._save_buffer_every > 0 and env_step % self._save_buffer_every == 0:
            self.buffer.save(f"{self.log_dir}/buffers/buffer_step_{env_step}.pickle")

    def _estimate_q(self, env_step: int):
        if self._estimate_q_every > 0 and env_step % self._estimate_q_every == 0:
            q_true = self.estimate_true_q()
            q_critic = self.estimate_critic_q()
            if q_true is not None:
                self._logger.log_scalar("trainer/Q-estimate", q_true, env_step)
                self._logger.log_scalar("trainer/Q-critic", q_critic, env_step)
                self._logger.log_scalar(
                    "trainer/Q_asb_diff", q_critic - q_true, env_step
                )

    def _log_stdout(self, env_step: int, batch):
        if env_step % self._stdout_log_every == 0:
            perc = int(env_step / self.num_steps * 100)
            print(
                f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {batch[2].mean():10.3f}"
            )

    def visualise_policy(self):
        """
        returned shape: [N, C, W, H]
        """
        env = self._make_env_test(seed=self.seed)
        try:
            imgs = []
            state, _ = env.reset()
            done = False
            while not done:
                img = env.render()
                imgs.append(img)
                action = self._algo.exploit(state)
                state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            return np.concatenate(imgs, dtype="uint8")
        except Exception as e:
            print(f"Failed to visualise a policy: {e}")
            return None

    def estimate_true_q(self, eval_episodes: int = 10) -> float | None:
        try:
            qs = []
            for i_eval in range(eval_episodes):
                env = self._make_env_test(seed=self.seed * 100 + i_eval)
                print("Before reset etimate q")
                state, _ = env.reset()

                q = 0
                s_i = 1
                while True:
                    action = self._algo.exploit(state)
                    state, r, terminated, truncated, _ = env.step(action)
                    q += r * self._gamma**s_i
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
            env = self._make_env_test(seed=self.seed * 100 + i_eval)

            state, _ = env.reset()
            action = self._algo.exploit(state)

            state = torch.tensor(state).unsqueeze(0).float().to(self._device)
            action = torch.tensor(action).unsqueeze(0).float().to(self._device)

            q = self._algo.critic(state, action)
            # TODO: TQC is not supported by this logic, need to update
            if isinstance(q, tuple):
                q = q[0]
            q = q.item()
            qs.append(q)

        return np.mean(qs, dtype=float)


def run_training(make_algo, make_env, make_logger, config: dict[str, Any], seed: int):
    env = make_env(seed=seed)
    logger = make_logger(seed)

    trainer = BaseTrainer(
        state_dim=config["state_shape"],
        action_dim=config["action_shape"],
        env=env,
        make_env_test=make_env,
        algo=make_algo(logger, seed),
        num_steps=config["num_steps"],
        eval_interval=config["eval_every"],
        device=config["device"],
        save_buffer_every=config["save_buffer"],
        visualise_every=config["visualise_every"],
        estimate_q_every=config["estimate_q_every"],
        stdout_log_every=config["log_every"],
        seed=seed,
        logger=logger,
    )

    trainer.train()
