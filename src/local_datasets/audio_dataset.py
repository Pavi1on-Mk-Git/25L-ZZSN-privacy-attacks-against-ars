import os
import os.path
from torch.utils.data import Dataset
import torch
import pandas as pd
from audiocraft.data.audio import audio_read


# @TODO: perhaps add partitioning per gpu like in the image dataset, though number of GPUs is set to 1 anyway
class AudioDataset(Dataset):
    def __init__(self, dataset_cfg):
        audio_dir = dataset_cfg.audio_dir
        labels_csv = dataset_cfg.labels_csv

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

        return audio_read(filename)[0], caption


def collate_fn(batch):
    # @TODO: can't torch.stack as the audios have different sizes
    #        verify if attacks will work with list or padded tensors
    #        using batch size of 1 only for now
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])
