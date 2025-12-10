import os
import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:

    def __init__(self, save_dir: str):

        self.save_log = os.path.join(save_dir, "log")
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'MeanReward':>15}"
                f"{'Loss':>15}{'PolicyLoss':>15}"
                f"{'Entropy':>15}{'ValueLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = os.path.join(save_dir, "reward_plot.jpg")
        self.ep_losses_plot = os.path.join(save_dir, "loss_plot.jpg")
        self.ep_policy_losses_plot = os.path.join(save_dir, "policy_loss_plot.jpg")
        self.ep_entropy_plot = os.path.join(save_dir, "entropy_plot.jpg")
        self.ep_value_loss_plot = os.path.join(save_dir, "value_loss_plot.jpg")

        # History metrics
        self.ep_rewards = []
        self.ep_losses = []
        self.ep_policy_losses = []
        self.ep_entropy = []
        self.ep_value_loss = []

        # Timing
        self.record_time = time.time()

    def log_episode(self, avg_reward, stats):
        self.ep_rewards.append(avg_reward)
        self.ep_losses.append(stats['loss'])
        self.ep_policy_losses.append(stats['policy_loss'])
        self.ep_entropy.append(stats['entropy'])
        self.ep_value_loss.append(stats['value_loss'])

    def record(self, episode):
        mean_ep_reward = np.round(self.ep_rewards[-1], 3)
        mean_ep_loss = np.round(self.ep_losses[-1], 3)
        mean_ep_policy_loss = np.round(self.ep_policy_losses[-1], 3)
        mean_ep_entropy = np.round(self.ep_entropy[-1], 3)
        ep_value_loss = np.round(self.ep_value_loss[-1], 3)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Loss {mean_ep_loss} - "
            f"PolicyLoss {mean_ep_policy_loss} - "
            f"Entropy {mean_ep_entropy} - "
            f"ValueLoss {ep_value_loss} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}"
                f"{mean_ep_reward:15.3f}{mean_ep_loss:15.3f}{mean_ep_policy_loss:15.3f}"
                f"{mean_ep_entropy:15.3f}{ep_value_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["losses", "policy_losses", "rewards", "entropy", "value_loss"]:
            plt.clf()
            plt.plot(getattr(self, f"ep_{metric}"), label=f"ep_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"ep_{metric}_plot"))
            plt.close()
