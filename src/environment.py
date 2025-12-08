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
                 angle_limit: Decimal = Decimal('180')):

        self.n_trees = n_trees
        self.placed_trees: list[ChristmasTree] = []
        self.scale_factor = scale_factor
        self.limit = limit
        self.angle_limit = angle_limit
        self.current_score = None
        self.step_count = 0
        self.obs_dim = 4 + self.n_trees * 5

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
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

    def _get_next_state(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        n_trees = len(self.placed_trees)
        if n_trees == 0:
            return obs

        n_norm = n_trees / self.n_trees

        # Compute bbox in *world units* ([-100,100]), not scaled polygons
        xmin, xmax, ymin, ymax = self._compute_bbox_world()
        w = xmax - xmin
        h = ymax - ymin

        # Max possible span is ~2 * COORD_LIMIT (from -100 to +100)
        L_max = float(2 * self.limit)
        bbox_w_norm = 0.0 if L_max == 0 else float(w) / L_max
        bbox_h_norm = 0.0 if L_max == 0 else float(h) / L_max

        step_norm = min(self.step_count / self.n_trees, 1.0)

        idx = 0
        obs[idx] = n_norm;      idx += 1
        obs[idx] = bbox_w_norm; idx += 1
        obs[idx] = bbox_h_norm; idx += 1
        obs[idx] = step_norm;   idx += 1

        # Per-tree features
        for i in range(self.n_trees):
            if i < n_trees:
                t = self.placed_trees[i]
                # Normalize positions back to [-1, 1]
                x_norm = float(t.center_x / self.limit)   # Decimal / Decimal -> Decimal
                y_norm = float(t.center_y / self.limit)

                angle_rad = np.deg2rad(float(t.angle))     # angle in radians
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

    def step(self, action):

        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (3,)

        ax, ay, a_theta = np.clip(action, -1.0, 1.0)

        # Map [-1, 1] -> [-COORD_LIMIT, COORD_LIMIT]
        x_tree = Decimal(str(ax)) * self.limit
        y_tree = Decimal(str(ay)) * self.limit

        # Map [-1, 1] -> [-angle_limit, angle_limit] degrees
        angle = Decimal(str(a_theta)) * self.angle_limit

        new_tree = ChristmasTree(
            center_x=x_tree,
            center_y=y_tree,
            angle=angle,
            scale_factor=self.scale_factor
        )

        # --- Collision check (safe when empty) ---
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

        # --- Reward hyperparams (start here, tune later) ---
        PLACE_BONUS = 100.0          # make "more trees" unequivocally good
        COLLISION_PENALTY = 50.0     # soft failure
        OOB_PENALTY = 50.0           # soft failure
        STEP_PENALTY = 0.1          # tiny time pressure
        SIDE_COEF = 0.4  # start here
        DELTA_SIDE_COEF = 0.2  # optional

        # area shaping coefficient:
        # Max bbox area ~ (200 * 200) = 40,000
        # So 0.05 * 1000 area delta would be 50 reward impact
        AREA_COEF = 0.05

        # old_area = self._get_bbox_area_world()
        old_side = self._get_bbox_side_world()

        if collision_found:
            # Fail-soft: don't terminate, just penalize and ignore placement
            reward -= COLLISION_PENALTY

        else:
            # Tentatively place
            self.placed_trees.append(new_tree)

            if self._has_exploded():
                # Soft penalty and undo the placement
                self.placed_trees.pop()
                reward -= OOB_PENALTY
            else:
                # Dense geometric shaping: punish increases in bounding area
                # new_area = self._get_bbox_area_world()
                # area_delta = max(new_area - old_area, 0.0)
                new_side = self._get_bbox_side_world()

                reward += PLACE_BONUS
                reward -= SIDE_COEF * new_side
                reward -= DELTA_SIDE_COEF * max(new_side - old_side, 0.0)
                # reward -= 0.1

                # Big positive for valid placement
                # reward += PLACE_BONUS
                #
                # # Small penalty for expanding footprint
                # reward -= AREA_COEF * area_delta

                # Keep score only for info/debug
                try:
                    self.current_score = self._get_current_score()
                except Exception:
                    # Shouldn't happen if we guard collisions, but keep safe
                    self.current_score = None

        # Small step penalty always
        reward -= STEP_PENALTY

        # Update step count
        self.step_count += 1

        # Success condition: placed all trees
        if len(self.placed_trees) == self.n_trees:
            # terminated = True
            # # Simple completion bonus
            # reward += 200.0
            terminated = True
            final_score = float(self.current_score) if self.current_score is not None else 1e9
            reward += 2000.0 / (1.0 + final_score)

        if len(self.placed_trees) < 5:
            # penalize absolute distance from origin early
            r = float((new_tree.center_x ** 2 + new_tree.center_y ** 2).sqrt())
            reward -= 0.2 * r

        # Hard cap attempts to avoid endless episodes
        # This makes training stable and aligns with “attempt N placements”
        if self.step_count >= self.n_trees and not terminated:
            truncated = True

        observation = self._get_next_state()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
