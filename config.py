import os
from decimal import Decimal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

print(os.environ)

class Settings(BaseSettings):
    N_TREES: int = int(os.getenv('N_TREES', '200'))
    EPOCHS: int = int(os.getenv('EPOCHS', '1000'))
    STEPS_PER_EPOCHS: int = int(os.getenv('STEPS_PER_EPOCHS', '2048'))
    SCALE_FACTOR: str = os.getenv('SCALE_FACTOR', '1e15')
    SCALE_LIMIT: str = os.getenv('SCALE_LIMIT', '100')
    ANGLE_LIMIT: str = os.getenv('ANGLE_LIMIT', '180')
    LOAD_WEIGHTS: bool = bool(int(os.getenv('LOAD_WEIGHTS', '0')))

    SAVE_DIR: str = os.getenv('SAVE_DIR', 'checkpoints')

    scale_factor: Decimal = Decimal(SCALE_FACTOR)
    limit: Decimal = Decimal(SCALE_LIMIT)
    angle_limit: Decimal = Decimal(ANGLE_LIMIT)

    root_log_dir: str = os.path.join("Logs_Checkpoints", "Model_logs")
    root_checkpoint_dir: str = os.path.join("Logs_Checkpoints", "Model_checkpoints")

    # Current log and checkpoint version directory
    log_dir: str = "version_0"
    checkpoint_dir: str = "version_0"

settings = Settings()