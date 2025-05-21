import os
from torch.utils.data import Dataset
import pandas as pd
from audiocraft.data.audio import audio_read


# @TODO: perhaps add partitioning per gpu like in the image dataset, though number of GPUs is set to 1 anyway
class AudioDataset(Dataset):
    def __init__(self, dataset_cfg):
        audio_dir = dataset_cfg.audio_dir
        labels_csv = dataset_cfg.labels_csv

        filenames = [filename for filename in os.listdir(audio_dir) if filename[-4:] == ".wav"]
        self.filenames = sorted(filenames, key=lambda name: int(name[:-4]))
        self.descriptions = pd.read_csv(labels_csv)["caption"].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        caption_idx = filename[:-4]
        caption = self.descriptions[caption_idx]

        return audio_read(filename)[0], caption
