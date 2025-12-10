import os
from decimal import Decimal
import gymnasium as gym
import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.ops import unary_union
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from .model import ChristmasTreeNet

class TreePackerLearner:

    def __init__(self, n_trees: int,
                 save_dir: str,
                 model_out_dir: str,
                 env: gym.Env,
                 steps_per_epochs: int = 2048,
                 load_weights: bool = False):

        self.n_trees = n_trees
        self.save_dir = save_dir
        self.model_out_path = os.path.join(model_out_dir, "christmas_tree_net.chkpt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.device}")
        self.env = env
        self.steps_per_epochs = steps_per_epochs

        self.batch_size = 1
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.1
        self.entropy_coef = 0.001
        self.value_coef = 0.2
        self.ppo_epochs = 100
        self.episode = 0
        self.best_score = np.inf

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.loss_fn = nn.MSELoss()

        self.net = ChristmasTreeNet(self.n_trees).float()
        self.net = self.net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        if load_weights:
            self.load()

    def load(self):
        self.net.load_state_dict(torch.load(self.model_out_path, weights_only=True))

    def save(self):
        save_path = self.model_out_path
        torch.save(self.net.state_dict(), save_path)
        print(f"ChristmasTreeNet saved to {save_path} at episode {self.episode}")

    def rollout(self):

        obs_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []

        obs, info = self.env.reset()
        done = False

        for _ in range(self.steps_per_epochs):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                mean, log_std, value = self.net(obs_tensor)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store transition
            obs_list.append(obs)
            actions_list.append(action_np)
            log_probs_list.append(log_prob.item())
            values_list.append(value.item())
            rewards_list.append(reward)
            dones_list.append(done)

            obs = next_obs
            if done:
                obs, info = self.env.reset()
                done = False

        # Bootstrap value for last state
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.net(obs_tensor)

        last_value = last_value.item()

        self.cache(state=np.array(obs_list, dtype=np.float32),
                   action=np.array(actions_list, dtype=np.float32),
                   log_prob=np.array(log_probs_list, dtype=np.float32),
                   value=np.array(values_list, dtype=np.float32),
                   reward=np.array(rewards_list, dtype=np.float32),
                   done=np.array(dones_list, dtype=np.bool_),
                   last_value=last_value
                   )

        return self.recall()

    def update_step(self, batch):

        obs, actions, old_log_probs, values, last_value, rewards, dones = batch
        # Compute GAE & returns
        advantages, returns = self.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value
        )

        # Convert to tensors
        obs_t = obs.to(self.device).float()
        actions_t = actions.to(self.device).float()
        old_log_probs_t = old_log_probs.to(self.device).float()
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            mean, log_std, values_pred = self.net(obs_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)  # (T,)
            entropy = dist.entropy().sum(dim=-1)  # (T,)

            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs_t)  # (T,)

            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values_pred = values_pred.squeeze(-1)  # (T,)
            value_loss = self.loss_fn(values_pred, returns_t)

            # Entropy loss
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "loss": loss.item(),
        }

    def compute_gae(self, rewards, values, dones, last_value):
        """
        rewards: [T]
        values:  [T] array of V(s_t)
        dones:   [T] bools
        last_value: scalar V(s_T) for bootstrap
        """
        T = len(rewards)
        values = np.append(values, last_value)  # now length T+1

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])  # 0 if done, 1 otherwise
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns

    def cache(self, state, action, log_prob, value, reward, done, last_value):

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()

        state = torch.tensor(state)
        action = torch.tensor(action)
        log_prob = torch.tensor(log_prob)
        value = torch.tensor(value)
        reward = torch.tensor(reward)
        done = torch.tensor(done)

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "action": action, "log_prob": log_prob, "value": value, "last_value": last_value, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        # batch = self.memory.sample(self.batch_size).to(self.device)
        n = len(self.memory)
        bs = min(self.batch_size, n)

        # indices for the most recent transitions
        idx = torch.arange(n - bs, n)

        # Most TorchRL buffers support indexing like this:
        batch = self.memory[idx].to(self.device)
        state, action, log_prob, value, last_value, reward, done = (batch.get(key) for key in ("state", "action", "log_prob", "value", "last_value", "reward", "done"))
        return state.squeeze(), action.squeeze(), log_prob.squeeze(), value.squeeze(), last_value.squeeze().item(), reward.squeeze(), done.squeeze()

    def train(self, n_episodes, logger):

        print("Artifacts will be saved to:", self.save_dir)

        tree_logger = logger(self.save_dir)

        iterator = tqdm(range(1, n_episodes + 1), dynamic_ncols=True)

        for e in iterator:

            self.episode = e

            batch = self.rollout()

            stats = self.update_step(batch)

            avg_reward = batch[5].mean()

            status = f"[Train][{e}] avg_rew={avg_reward:.3f}, "
            status+= f"loss={stats['loss']:.3f}, "
            status+= f"policy_loss={stats['policy_loss']:.3f}, "
            status+= f"value_loss={stats['value_loss']:.3f}, "
            status+= f"entropy={stats['entropy']:.3f}"

            iterator.set_description(status)

            tree_logger.log_episode(avg_reward, stats)

            if (e % 5 == 0) or (e == n_episodes):
                self.place_trees()
                current_score = self.env.unwrapped._get_current_score()
                print(f"Episode {e}: Current Score: {float(current_score):.12f}, Best Score: {float(self.best_score):.12f}")
                if current_score < self.best_score:
                    self.save()
                    self.best_score = current_score
                    self.save_plot_results(Decimal(self.env.unwrapped._get_side_length()),
                                           self.env.unwrapped.placed_trees,
                                           len(self.env.unwrapped.placed_trees),
                                           self.save_dir,
                                           current_score)

            tree_logger.record(e)

    def place_trees(self):
        obs, info = self.env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                mean, log_std, value = self.net(obs_tensor)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            obs_tensor = next_obs

    def save_plot_results(self, side_length, placed_trees, num_trees, save_dir, score):
        scale_factor = self.env.unwrapped.scale_factor

        _, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

        all_polygons = [t.polygon for t in placed_trees]
        bounds = unary_union(all_polygons).bounds

        for i, tree in enumerate(placed_trees):
            # Rescale for plotting
            x_scaled, y_scaled = tree.polygon.exterior.xy
            x = [Decimal(val) / scale_factor for val in x_scaled]
            y = [Decimal(val) / scale_factor for val in y_scaled]
            ax.plot(x, y, color=colors[i])
            ax.fill(x, y, alpha=0.5, color=colors[i])

        minx = Decimal(bounds[0]) / scale_factor
        miny = Decimal(bounds[1]) / scale_factor
        maxx = Decimal(bounds[2]) / scale_factor
        maxy = Decimal(bounds[3]) / scale_factor

        width = maxx - minx
        height = maxy - miny

        square_x = minx if width >= height else minx - (side_length - width) / 2
        square_y = miny if height >= width else miny - (side_length - height) / 2
        bounding_square = Rectangle(
            (float(square_x), float(square_y)),
            float(side_length),
            float(side_length),
            fill=False,
            edgecolor='red',
            linewidth=2,
            linestyle='--',
        )
        ax.add_patch(bounding_square)

        padding = 0.5
        ax.set_xlim(
            float(square_x - Decimal(str(padding))),
            float(square_x + side_length + Decimal(str(padding))))
        ax.set_ylim(float(square_y - Decimal(str(padding))),
                    float(square_y + side_length + Decimal(str(padding))))
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.title(f'{num_trees} Trees: {side_length:.12f} Score: {float(score):.12f}')
        plt.savefig(os.path.join(save_dir, 'trees.png'))
        plt.close()
