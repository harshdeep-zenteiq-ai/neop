from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, GINO_Small3d
from .opt import PatchingConfig
from .wandb import WandbConfig


class CarCFDDatasetConfig(ConfigBase):
    root: str = "~/data/car-pressure-data/processed-car-pressure-data"
    sdf_query_resolution: int = 32
    n_train: int = 250 #500
    n_test: int = 50#111
    download: bool = True

# FileNotFoundError: [Errno 2] No such file or directory: '/home/deepan/data/car-pressure-data/processed-car-pressure-data/train.txt'

class CarCFDOptConfig(ConfigBase):
    n_epochs: int = 301 #301
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = GINO_Small3d()
    opt: ConfigBase = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()  # default empty
