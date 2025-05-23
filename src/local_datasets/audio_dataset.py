import os
import os.path
from torch.utils.data import Dataset
import torch
from torch.types import Tensor as T
import pandas as pd
from audiocraft.data.audio import audio_read


# @TODO: perhaps add partitioning per gpu like in the image dataset, though number of GPUs is set to 1 anyway
class AudioDataset(Dataset):
    def __init__(self, dataset_cfg):
        if dataset_cfg.split == "train":
            audio_dir = dataset_cfg.train_audio_dir
            labels_csv = dataset_cfg.train_labels_csv
        elif dataset_cfg.split == "test":
            audio_dir = dataset_cfg.test_audio_dir
            labels_csv = dataset_cfg.test_labels_csv
        else:
            raise Exception(f"Invalid dataset split: {dataset_cfg.split}; expected 'train' or 'test'")

        filenames = [filename for filename in os.listdir(audio_dir) if filename[-4:] == ".wav"]
        filenames = sorted(filenames, key=lambda name: int(name[:-4]))
        self.filenames = [os.path.join(audio_dir, filename) for filename in filenames]
        self.descriptions = pd.read_csv(labels_csv)["caption"].tolist()

        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        caption_idx = int(filename[:-4].split("/")[-1])
        caption = self.descriptions[caption_idx]

        return self._read_audio_as_mono(filename), caption

    def _read_audio_as_mono(self, filename: str) -> T:
        audio = audio_read(filename)[0]

        if audio.shape[0] == 2:
            channel_1, channel_2 = audio[0], audio[1]
            audio = torch.unsqueeze((channel_1 + channel_2) / 2, 0)

        return audio


def collate_fn(batch):
    # can't torch.stack as the audios have different sizes, using batch size of 1 only for now
    # @TODO: add proper batching, with returning padding mask to be used by tokenizer
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])
