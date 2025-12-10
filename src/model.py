import torch
from torch import nn
import torch.nn.functional as F

class ChristmasTreeNet(nn.Module):

    def __init__(self,
                 n_trees: int,
                 action_dim: int = 4,
                 tree_hidden_dim: int = 32,
                 trunk_hidden_dim: int = 256):
        super().__init__()

        self.n_trees = n_trees
        self.tree_feat_dim = 5   # [x_norm, y_norm, cosθ, sinθ, present]
        self.global_dim = 4      # [n_norm, bbox_w_norm, bbox_h_norm, step_norm]
        self.tree_hidden_dim = tree_hidden_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.input_dim = self.global_dim + self.n_trees * self.tree_feat_dim
        self.output_dim = action_dim

        # ----- Per-tree encoder: shared MLP over each tree -----
        # Input: 5-dim per-tree feature
        # Output: tree_hidden_dim-dim embedding
        self.tree_mlp = nn.Sequential(
            nn.Linear(4, self.tree_hidden_dim),  # ← change 5 → 4
            nn.ReLU(),
            nn.Linear(self.tree_hidden_dim, self.tree_hidden_dim),
            nn.ReLU(),
        )

        # ----- Trunk MLP over [globals || pooled_tree_embedding] -----
        # Input dim = 4 globals + tree_hidden_dim
        trunk_input_dim = self.global_dim + self.tree_hidden_dim

        self.trunk_fc1 = nn.Linear(trunk_input_dim, self.trunk_hidden_dim)
        self.trunk_fc2 = nn.Linear(self.trunk_hidden_dim, self.trunk_hidden_dim)

        # Actor head
        self.actor_mean = nn.Linear(self.trunk_hidden_dim, self.output_dim)
        self.log_std = nn.Parameter(torch.ones(self.output_dim) * -1.0)

        # Critic head
        self.critic = nn.Linear(self.trunk_hidden_dim, 1)


    def forward(self, obs: torch.Tensor):
        """
        obs: (batch_size, obs_dim)
        returns:
          - mean: (batch_size, action_dim), in [-1, 1] via tanh
          - log_std: (batch_size, action_dim)
          - value: (batch_size, 1)
        """
        batch_size = obs.shape[0]

        # ----- Split globals and per-tree features -----
        globals_feat = obs[:, : self.global_dim]  # (B, 4)
        tree_feats_flat = obs[:, self.global_dim :]  # (B, n_trees * 5)

        # Reshape into (B, n_trees, 5)
        tree_feats = tree_feats_flat.reshape(
            batch_size, self.n_trees, self.tree_feat_dim
        )  # (B, T, 5)

        # Last feature is 'present' mask in [0,1]
        present_mask = tree_feats[..., -1]  # (B, T)

        # Features to encode (exclude present): x,y,cosθ,sinθ
        tree_core = tree_feats[..., :4]  # (B, T, 4)

        # ----- Encode each tree independently with shared MLP -----
        # Flatten for MLP: (B*T, 4) -> (B*T, tree_hidden_dim)
        tree_core_flat = tree_core.reshape(batch_size * self.n_trees, 4)
        tree_emb_flat = self.tree_mlp(tree_core_flat)
        tree_emb = tree_emb_flat.reshape(batch_size, self.n_trees, -1)  # (B, T, H)

        # ----- Mask and pool over trees (mean over present ones) -----
        # Expand mask to match embedding dims: (B, T, 1)
        mask = present_mask.unsqueeze(-1)  # (B, T, 1)

        # Zero-out embeddings where no tree is present
        tree_emb_masked = tree_emb * mask  # (B, T, H)

        # Sum over trees
        emb_sum = tree_emb_masked.sum(dim=1)  # (B, H)

        # Number of present trees per batch (avoid division by 0)
        count = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)

        # Mean pooling
        tree_emb_pooled = emb_sum / count  # (B, H)

        # ----- Concatenate globals + pooled tree embedding -----
        trunk_input = torch.cat([globals_feat, tree_emb_pooled], dim=-1)  # (B, 4+H)

        # ----- Trunk -----
        x = F.relu(self.trunk_fc1(trunk_input))
        x = F.relu(self.trunk_fc2(x))

        # ----- Actor -----
        mean = torch.tanh(self.actor_mean(x))  # (B, action_dim) in [-1, 1]
        log_std = self.log_std.expand_as(mean)  # broadcast to (B, action_dim)

        # ----- Critic -----
        value = self.critic(x)  # (B, 1)

        return mean, log_std, value
