import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from figaro.datasets import MidiDataset, SeqCollator
from pathlib import Path


def get_figaro_dataloader(config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig):
    data_dir = Path(dataset_cfg.data_dir)
    match dataset_cfg.split:
        case "train":
            split_files_data = Path(dataset_cfg.train_data)
        case "test":
            split_files_data = Path(dataset_cfg.test_data)
        case _:
            raise Exception(f"Invalid dataset split: {dataset_cfg.split}; expected 'train' or 'test'")

    with split_files_data.open("r") as f:
        split_paths = [line.strip() for line in f if line.strip()]

    files = []
    for rel_path in split_paths:
        file_path = data_dir / rel_path
        files.append(str(file_path))

    dataset = MidiDataset(midi_files=files, max_len=256, max_bars=256, description_flavor="description")
    coll = SeqCollator(context_size=-1)

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        generator=g,
        collate_fn=coll,
    )
