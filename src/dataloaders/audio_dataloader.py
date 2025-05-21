import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.local_datasets import datasets


def get_audio_dataloader(config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig):
    dataset = datasets[dataset_cfg.name](dataset_cfg)

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=1,  # audio unbatched
        shuffle=True,
        generator=g,
        collate_fn=dataset.collate_fn,
    )
