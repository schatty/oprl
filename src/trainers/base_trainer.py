import numpy as np

from trainers.buffers.episodic_buffer import EpisodicReplayBuffer


class BaseTrainer:

    def __init__(self, state_shape=None, action_shape=None, env=None, make_env_test=None, algo=None, buffer_size=int(1e6),
                 gamma=0.99, device=None, num_steps=int(1e6), start_steps=int(10e3), batch_size=128,
                 eval_interval=int(2e3), num_eval_episodes=10, save_buffer_every=0, visualize_every=0,
                 estimate_q_every=0, stdout_log_every=int(1e5), seed=0, logger=None):
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
            visualize_every: Number of env steps after which perform vizualization.
            stdout_log_every: Number of evn steps after which log info to stdout.
            seed: Random seed.
        """
        self.env = env
        self.make_env_test = make_env_test
        self.algo = algo
        self.gamma = gamma
        self.device = device
        self.save_buffer_every = save_buffer_every
        self.visualize_every = visualize_every
        self.estimate_q_every = estimate_q_every
        self.stdout_log_every = stdout_log_every
        self.logger = logger
        self.seed = seed

        self.buffer = EpisodicReplayBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
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
        mean_reward = 0
        state, _ = self.env.reset(seed=self.seed)

        for env_step in range(self.num_steps + 1):
            ep_step += 1
            if env_step <= self.start_steps:
                action = self.env.sample_action()
            else:
                action = self.algo.explore(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.buffer.append(state, action, reward, terminated, episode_done=terminated or truncated)
            if terminated or truncated:
                next_state, _ = self.env.reset(seed=self.seed)
                ep_step = 0
            state = next_state

            if len(self.buffer) < self.batch_size:
                continue
            batch = self.buffer.sample(self.batch_size)
            self.algo.update(batch)

            if env_step % self.eval_interval == 0:
                mean_reward = self.evaluate()
                self.logger.log_scalar("trainer/ep_reward", mean_reward, env_step)
                self.logger.log_scalar("trainer/avg_reward", batch[2].mean(), env_step)
                self.logger.log_scalar("trainer/buffer_transitions", len(self.buffer), env_step)
                self.logger.log_scalar("trainer/buffer_episodes", self.buffer.num_episodes, env_step)
                self.logger.log_scalar("trainer/buffer_last_ep_len", self.buffer.get_last_ep_len(), env_step)

            if self.visualize_every > 0 and env_step % self.visualize_every == 0:
                # imgs = self.visualize_policy()
                # if imgs is not None:
                #     wandb.log({"video": wandb.Video(imgs, fps=25, format="gif"), "env_step": env_step})
                pass

            if self.save_buffer_every > 0 and env_step % self.save_buffer_every == 0:
                self.buffer.save(f"{self.log_dir}/buffers/buffer_step_{env_step}.pickle")

            if self.estimate_q_every > 0 and env_step % self.estimate_q_every == 0:
                q_est = self.estimate_true_q()
                q_critic = self.estimate_critic_q()
                if q_est is not None:
                    self.logger.log_scalar("trainer/Q-estimate", q_est,  env_step)
                    self.logger.log_scalar("trainer/Q-critic", q_critic, env_step)

            if env_step % self.stdout_log_every == 0:
                perc = int(env_step / self.num_steps * 100)
                print(f"Env step {env_step:8d} ({perc:2d}%) Avg Reward {batch[2].mean():10.3f}"
                      f"Ep Reward {mean_reward:10.3f}")

    def evaluate(self):
        returns = []
        for i_ep in range(self.num_eval_episodes):
            env_test = self.make_env_test(seed=self.seed + i_ep)
            state, _ = env_test.reset(seed=self.seed + i_ep)

            episode_return = 0.0
            terminated, truncated = False, False

            while not (terminated or truncated):
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        return mean_return

    def visualize_policy(self):
        try:
            imgs = []
            state, _ = self.env_test.reset(seed=self.seed)
            done = False
            while not done:
                img = self.env_test.render(mode="rgb_array")
                img = np.moveaxis(img, (0, 1, 2), (1, 2, 0))
                img = np.expand_dims(img, 0)
                imgs.append(img)
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)
            return np.concatenate(imgs)
        except Exception as e:
            print(f"Failed to visualize a policy: {e}")
            return None

    def estimate_true_q(self, eval_episodes=10):
        """Estimates true Q-value via launching given policy from sampled state until
        the end of an episode. """

        # TODO: Works only for some envs of OpenAI
        try:
            qs = []
            for _ in range(eval_episodes):
                # TODO: add seed
                self.env_test.reset()

                states, _, rewards, _, _ = self.buffer.sample(1)
                state, reward = states[0].cpu().numpy(), rewards[0].cpu().numpy()

                qpos = state[:self.env_test.model.nq - 1]
                qvel = state[self.env_test.model.nq - 1:]
                qpos = np.concatenate([[0], qpos])

                self.env_test.set_state(qpos, qvel)

                q = reward
                s_i = 1
                while True:
                    action = self.algo.exploit(np.array(state))
                    state, r, d, _ = self.env_test.step(action)
                    q += r * self.gamma ** s_i
                    if d or s_i == self.env_test._max_episode_steps:
                        break
                    s_i += 1
                qs.append(q)

            return np.mean(qs)
        except Exception as e:
            print(f"Failed to estimate Q-value: {e}")
            return None

    def estimate_critic_q(self, num_samples=100):
        states, actions, _, _, _ = self.buffer.sample(num_samples)
        q = self.algo.critic(states, actions)
        if isinstance(q, (list, tuple)):
            q = q[0]
        return q.detach().mean().item()
