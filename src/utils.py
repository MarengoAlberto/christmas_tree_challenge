from decimal import Decimal
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

def setup_log_directory(training_config):
    '''Tensorboard Log and Model checkpoint directory Setup'''

    if os.path.isdir(training_config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(training_config.root_log_dir)]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = training_config.log_dir

    # Update the training config default directory
    training_config.log_dir        = os.path.join(training_config.root_log_dir,        version_name)
    training_config.checkpoint_dir = os.path.join(training_config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(training_config.log_dir,        exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    print(f"Logging at: {training_config.log_dir}")
    print(f"Model Checkpoint at: {training_config.checkpoint_dir}")

    return training_config, version_name

def plot_results(side_length, placed_trees, num_trees, scale_factor):
    """Plots the arrangement of trees and the bounding square."""
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
    plt.title(f'{num_trees} Trees: {side_length:.12f}')
    plt.show()
    plt.close()

def world_to_action(x, y, deg, coord_limit=100.0, angle_limit_deg=180.0):
    ax = np.clip(x / coord_limit, -1.0, 1.0)
    ay = np.clip(y / coord_limit, -1.0, 1.0)
    a_theta = np.clip(deg / angle_limit_deg, -1.0, 1.0)
    return np.array([ax, ay, a_theta], dtype=np.float32)

def get_world_values(df, idx):
    x = float(df.reset_index(drop=True).loc[idx, ['x']].values[0][1:])
    y = float(df.reset_index(drop=True).loc[idx, ['y']].values[0][1:])
    deg = float(df.reset_index(drop=True).loc[idx, ['deg']].values[0][1:])
    return x, y, deg

def get_random_action(COORD_LIMIT, ANGLE_LIMIT_DEG):
    x = float(random.randint(-int(COORD_LIMIT), int(COORD_LIMIT)))
    y = float(random.randint(-int(COORD_LIMIT), int(COORD_LIMIT)))
    deg = float(random.randint(-int(ANGLE_LIMIT_DEG), int(ANGLE_LIMIT_DEG)))
    return x,y,deg
