import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src import ChristmasTreePacker, TreePackerLearner, MetricLogger, setup_log_directory
from config import settings

print("Registering ChristmasTreePacker environment...")
gym.register(
    id="gymnasium_env/ChristmasTreePacker-v0",
    entry_point=ChristmasTreePacker,
    max_episode_steps=300,  # Prevent infinite episodes
)
print("Environment registered successfully.")
print("Creating environment instance...")
env = gym.make("gymnasium_env/ChristmasTreePacker-v0", n_trees=settings.N_TREES)

try:
    raw_env = env.unwrapped
    check_env(raw_env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
    raise ValueError("Environment failed the checks.")

def run():
    config, version_name = setup_log_directory(settings)
    print(f"Config setup complete for version: {version_name}")
    print(f"Config: {config}")
    learner = TreePackerLearner(config.N_TREES,
                                config.log_dir,
                                config.checkpoint_dir,
                                env,
                                steps_per_epochs=config.STEPS_PER_EPOCHS,
                                load_weights=config.LOAD_WEIGHTS)

    print("Starting training...")
    learner.train(config.EPOCHS, MetricLogger)
    print("Training completed.")


if __name__ == "__main__":
    run()
