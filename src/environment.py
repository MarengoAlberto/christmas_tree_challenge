import gymnasium as gym
from typing import Optional
from decimal import Decimal
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .core import ChristmasTree

class ChristmasTreePacker(gym.Env):

    def __init__(self,
                 n_trees: int,
                 scale_factor: Decimal = Decimal('1e15'),
                 limit: int = Decimal('100'),
                 angle_limit: Decimal = Decimal('180'),
                 global_warmup: int = 10,          # <--- NEW
                 min_local_radius: float = 0.5,    # <--- NEW (world units)
                 max_local_radius: float = 20.0):  # <--- NEW

        self.n_trees = n_trees
        self.placed_trees: list[ChristmasTree] = []
        self.scale_factor = scale_factor
        self.limit = limit
        self.angle_limit = angle_limit
        self.current_score = None
        self.step_count = 0

        # NEW
        self.global_warmup = global_warmup
        self.min_local_radius = float(min_local_radius)
        self.max_local_radius = float(max_local_radius)

        # If you keep obs as-is, no change needed here
        self.obs_dim = 4 + self.n_trees * 5

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # NEW 4D action:
        # a0 = "anchor selector" (ignored during global warmup)
        # a1,a2 = dx,dy offsets in local frame
        # a3 = rotation
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )


    def _get_side_length(self):
        if not self.placed_trees:
            return 0.0

        all_polygons = [t.polygon for t in self.placed_trees]
        bounds = unary_union(all_polygons).bounds

        minx = Decimal(bounds[0]) / self.scale_factor
        miny = Decimal(bounds[1]) / self.scale_factor
        maxx = Decimal(bounds[2]) / self.scale_factor
        maxy = Decimal(bounds[3]) / self.scale_factor

        width = max(maxx - minx, 0.0)
        height = max(maxy - miny, 0.0)
        # this forces a square bounding using the largest side

        return float(max(width, height))

    def _get_info(self):
        return {
            "n_trees_packed": len(self.placed_trees),
            "side_length": self._get_side_length(),
            "current_score": self.current_score,
        }

    def _has_exploded(self):
        all_x = np.array([p.center_x for p in self.placed_trees])
        all_y = np.array([p.center_y for p in self.placed_trees])
        bad_x = (all_x.astype(float) < -self.limit).any() or (all_x.astype(float) > self.limit).any()
        bad_y = (all_y.astype(float) < -self.limit).any() or (all_y.astype(float) > self.limit).any()
        if bad_x or bad_y:
            return True
        return False

    def _get_bbox_side_world(self) -> float:
        if not self.placed_trees:
            return 0.0
        xmin, xmax, ymin, ymax = self._compute_bbox_world()
        w = max(xmax - xmin, Decimal("0"))
        h = max(ymax - ymin, Decimal("0"))
        return float(max(w, h))

    def _compute_bbox_world(self):
        all_polygons = [t.polygon for t in self.placed_trees]
        bounds = unary_union(all_polygons).bounds

        minx = Decimal(bounds[0]) / self.scale_factor
        miny = Decimal(bounds[1]) / self.scale_factor
        maxx = Decimal(bounds[2]) / self.scale_factor
        maxy = Decimal(bounds[3]) / self.scale_factor

        return minx, maxx, miny, maxy

    def _centroid(self):
        if not self.placed_trees:
            return Decimal("0"), Decimal("0")
        xs = [t.center_x for t in self.placed_trees]
        ys = [t.center_y for t in self.placed_trees]
        return sum(xs) / Decimal(len(xs)), sum(ys) / Decimal(len(ys))

    def _local_radius(self):
        # start wider, end tighter
        n = len(self.placed_trees)
        n_norm = n / self.n_trees if self.n_trees else 0.0
        r = 20.0 * (1.0 - n_norm) + 3.0  # world units
        return Decimal(str(r))

    def _anchor_index_from_action(self, a_anchor: float) -> int:
        n = len(self.placed_trees)
        if n == 0:
            return -1  # special case: no anchor

        # map [-1,1] -> [0, n-1]
        u = (a_anchor + 1.0) / 2.0
        idx = int(np.clip(np.floor(u * n), 0, n-1))
        return idx

    def _decode_action(self, action: np.ndarray):
        """
        Returns (x_tree, y_tree, angle) as Decimals.

        - During warmup: interpret as global placement using a1,a2,a3
        - After warmup: anchor-based local offsets using a0,a1,a2,a3
        """
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (4,)

        a_anchor, ax, ay, a_theta = np.clip(action, -1.0, 1.0)

        # Map rotation always the same
        angle = Decimal(str(a_theta)) * self.angle_limit

        # Global warmup phase
        if len(self.placed_trees) < self.global_warmup:
            x_tree = Decimal(str(ax)) * self.limit
            y_tree = Decimal(str(ay)) * self.limit
            return x_tree, y_tree, angle

        # Local phase
        idx = self._anchor_index_from_action(float(a_anchor))
        anchor = self.placed_trees[idx]

        r = self._local_radius()
        dx = Decimal(str(ax)) * self._local_radius()
        dy = Decimal(str(ay)) * self._local_radius()

        x_tree = anchor.center_x + dx
        y_tree = anchor.center_y + dy

        return x_tree, y_tree, angle


    def _get_next_state(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        n_trees = len(self.placed_trees)
        if n_trees == 0:
            return obs

        n_norm = n_trees / self.n_trees

        xmin, xmax, ymin, ymax = self._compute_bbox_world()
        w = xmax - xmin
        h = ymax - ymin

        L_max = float(2 * self.limit)
        bbox_w_norm = 0.0 if L_max == 0 else float(w) / L_max
        bbox_h_norm = 0.0 if L_max == 0 else float(h) / L_max

        # NEW: compactness proxy in [0,1-ish]
        # side / sqrt(n) normalized by L_max
        side = self._get_bbox_side_world()
        compact = 0.0
        if n_trees > 0 and L_max > 0:
            compact = (side / max(np.sqrt(n_trees), 1e-6)) / L_max

        idx = 0
        obs[idx] = float(n_norm);       idx += 1
        obs[idx] = float(bbox_w_norm);  idx += 1
        obs[idx] = float(bbox_h_norm);  idx += 1
        obs[idx] = float(compact);      idx += 1

        # Per-tree features unchanged
        for i in range(self.n_trees):
            if i < n_trees:
                t = self.placed_trees[i]
                x_norm = float(t.center_x / self.limit)
                y_norm = float(t.center_y / self.limit)
                angle_rad = np.deg2rad(float(t.angle))
                cos_th = np.cos(angle_rad)
                sin_th = np.sin(angle_rad)
                present = 1.0
            else:
                x_norm = y_norm = cos_th = sin_th = 0.0
                present = 0.0

            obs[idx] = x_norm;   idx += 1
            obs[idx] = y_norm;   idx += 1
            obs[idx] = cos_th;   idx += 1
            obs[idx] = sin_th;   idx += 1
            obs[idx] = present;  idx += 1

        return obs

    def _get_current_score(self):

        num_trees = len(self.placed_trees)

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in self.placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    raise ValueError(f'Overlapping trees')

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square boulding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (Decimal(side_length_scaled) ** 2) / (self.scale_factor**2) / Decimal(num_trees)
        return group_score

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        self.placed_trees = []
        self.step_count = 0
        self.current_score = None

        observation = self._get_next_state()
        info = self._get_info()

        return observation, info

    # def step(self, action):
    #
    #     action = np.asarray(action, dtype=np.float32)
    #     assert action.shape == (3,)
    #
    #     ax, ay, a_theta = np.clip(action, -1.0, 1.0)
    #
    #     # Map [-1, 1] -> [-COORD_LIMIT, COORD_LIMIT]
    #     x_tree = Decimal(str(ax)) * self.limit
    #     y_tree = Decimal(str(ay)) * self.limit
    #
    #     # Map [-1, 1] -> [-180, 180] degrees
    #     angle = Decimal(str(a_theta)) * self.angle_limit
    #
    #     new_tree = ChristmasTree(center_x=x_tree, center_y=y_tree,angle=angle, scale_factor=self.scale_factor)
    #     # new_tree.polygon = affinity.translate(
    #     #     new_tree.polygon,
    #     #     xoff=float(new_tree.center_x * self.scale_factor),
    #     #     yoff=float(new_tree.center_y * self.scale_factor),
    #     # )
    #
    #     placed_polygons = [p.polygon for p in self.placed_trees]
    #     tree_index = STRtree(placed_polygons)
    #
    #     collision_found = False
    #     # Looking for nearby objects
    #     possible_indices = tree_index.query(new_tree.polygon)
    #     # This is the collision detection step
    #     if any((new_tree.polygon.intersects(placed_polygons[i]) and not new_tree.polygon.touches(placed_polygons[i])) for i in possible_indices):
    #         collision_found = True
    #
    #     reward = 0.0
    #     terminated = False
    #     truncated = False
    #
    #     if collision_found:
    #         # Natural terminal failure
    #         reward = -5.0
    #         terminated = True
    #
    #     else:
    #         self.placed_trees.append(new_tree)
    #
    #         if self._has_exploded():
    #             # Also a natural terminal failure
    #             reward = -5.0
    #             terminated = True
    #
    #         else:
    #             old_score = self.current_score  # could be None at first
    #             current_score = self._get_current_score()  # lower is better
    #
    #             if old_score is None:
    #                 # First tree: don’t punish too much, no reference yet
    #                 step_reward = 0.0
    #             else:
    #                 # Reward = improvement in score
    #                 delta = float(old_score) - float(current_score)   # >0 if improved
    #                 step_reward = delta * 10.0          # scale factor
    #
    #             # Small step penalty to encourage fewer, better moves
    #             step_penalty = 0.01
    #             reward = step_reward - step_penalty
    #
    #             self.current_score = current_score
    #
    #             # If we placed all trees: terminal success
    #             if len(self.placed_trees) == self.n_trees:
    #                 terminated = True
    #                 # Add a finishing bonus based on how good the final score is
    #                 reward += 10.0 * (1.0 / (1.0 + current_score))
    #                 # (or something like -current_score * big_factor)
    #
    #     observation = self._get_next_state()
    #     info = self._get_info()
    #     self.step_count += 1
    #     return observation, reward, terminated, truncated, info

    def _get_bbox_area_world(self) -> float:
        """Area of the current bounding box in world units."""
        if not self.placed_trees:
            return 0.0
        xmin, xmax, ymin, ymax = self._compute_bbox_world()
        w = max(xmax - xmin, Decimal("0"))
        h = max(ymax - ymin, Decimal("0"))
        return float(w * h)

    def _edge_penalty(self, x, y):
        # x,y are Decimal world coords
        x = float(abs(x))
        y = float(abs(y))
        lim = float(self.limit)

        # penalty ramps up in outer 20% of space
        margin = 0.2 * lim
        px = max(0.0, x - (lim - margin)) / margin
        py = max(0.0, y - (lim - margin)) / margin
        return px + py

    def _compactness(self) -> float:
        n = max(len(self.placed_trees), 1)
        side = self._get_bbox_side_world()
        return side / np.sqrt(n)

    def _mean_nn_dist(self):
        if len(self.placed_trees) < 2:
            return 0.0
        pts = np.array([[float(t.center_x), float(t.center_y)] for t in self.placed_trees])
        # cheap O(n^2) is fine for 200
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        d += np.eye(len(pts)) * 1e9
        return float(d.min(axis=1).mean())

    def step(self, action):

        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (4,)

        # Decode global-or-local action
        x_tree, y_tree, angle = self._decode_action(action)

        new_tree = ChristmasTree(
            center_x=x_tree,
            center_y=y_tree,
            angle=angle,
            scale_factor=self.scale_factor
        )

        placed_polygons = [p.polygon for p in self.placed_trees]

        if placed_polygons:
            tree_index = STRtree(placed_polygons)
            possible_indices = tree_index.query(new_tree.polygon)
            collision_found = any(
                (new_tree.polygon.intersects(placed_polygons[i])
                 and not new_tree.polygon.touches(placed_polygons[i]))
                for i in possible_indices
            )
        else:
            collision_found = False

        reward = 0.0
        terminated = False
        truncated = False

        n = len(self.placed_trees)
        n_norm = n / self.n_trees

        # --- Hyperparams ---
        PLACE_BONUS = 5.0 if n < 50 else 2.0
        COLLISION_PENALTY = 1.0 if n < 50 else 3.0
        OOB_PENALTY = 1.0 if n < 50 else 3.0
        STEP_PENALTY = 0.01

        # Dense compactness shaping based on bbox side
        SHAPE_COEF = 2.0

        # Edge discouragement
        EDGE_COEF = 0.5 if n < 50 else 2.0

        MAX_ATTEMPTS = int(self.n_trees * 2.0)

        old_side = self._get_bbox_side_world()

        if collision_found:
            reward = -COLLISION_PENALTY

        else:
            # Tentatively place
            self.placed_trees.append(new_tree)

            if self._has_exploded():
                self.placed_trees.pop()
                reward = -OOB_PENALTY

            else:
                # Base reward
                reward = PLACE_BONUS

                # Dense side shrink/growth reward
                new_side = self._get_bbox_side_world()

                # Weight compactness more later in the episode
                n = len(self.placed_trees)
                n_norm = n / self.n_trees
                shape_weight = 1.0 + 3.0 * n_norm

                reward += shape_weight * SHAPE_COEF * (old_side - new_side)

                # Soft absolute footprint pressure
                L_max = float(2 * self.limit)
                if L_max > 0:
                    reward -= 0.2 * (new_side / L_max)

                # Edge penalty
                reward -= EDGE_COEF * self._edge_penalty(new_tree.center_x, new_tree.center_y)

                # Centroid Reward
                cx, cy = self._centroid()
                dist = float(((new_tree.center_x - cx) ** 2 + (new_tree.center_y - cy) ** 2).sqrt())
                reward -= 0.02 * dist

                # Update score for info only
                try:
                    self.current_score = self._get_current_score()
                except Exception:
                    self.current_score = None

        # Time pressure
        reward -= STEP_PENALTY

        self.step_count += 1

        # Success
        if len(self.placed_trees) == self.n_trees:
            terminated = True
            final_score = float(self.current_score) if self.current_score is not None else 1e9
            reward += 50.0 * (1.0 / (1.0 + final_score))

        # Attempt cap
        if self.step_count >= MAX_ATTEMPTS and not terminated:
            truncated = True
            missing = self.n_trees - len(self.placed_trees)
            reward -= min(10.0 * missing, 50.0)

        # Hard cap to avoid endless rollouts
        if self.step_count >= self.n_trees and not terminated:
            truncated = True

        observation = self._get_next_state()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
