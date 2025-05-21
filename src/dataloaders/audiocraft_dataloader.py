import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.local_datasets import datasets


def get_audiocraft_dataloader(config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig):
    dataset = datasets[dataset_cfg.name](dataset_cfg)
    dataset = dataset.dataset

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
