import os
from torch.utils.data import Dataset
import pandas as pd
import os
from audiocraft.data.audio import audio_read


# @TODO: add partitioning per gpu like in the image dataset
class AudioDataset(Dataset):
    def __init__(self, audio_dir: str, labels_csv: str):
        self.audio_dir = audio_dir
        self.labels_csv = labels_csv

        filenames = [filename for filename in os.listdir(audio_dir) if filename[-4:] == ".wav"]
        self.filenames = sorted(filenames, key=lambda name: int(name[:-4]))
        self.descriptions = pd.read_csv(labels_csv)["caption"].tolist()

        print(self.filenames[:3])
        print(self.descriptions[:3])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        caption_idx = filename[:-4]
        caption = self.descriptions[caption_idx]

        return audio_read(filename)[0], caption
