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

        if dataset_cfg.name == "musiccaps":
            self.descriptions = pd.read_csv(labels_csv)["caption"].tolist()
        elif dataset_cfg.name == "audiocaps":
            data = pd.read_csv(labels_csv)
            indices, captions = data["audiocap_id"].tolist(), data["caption"].tolist()
            self.descriptions = {index: caption for index, caption in zip(indices, captions)}
        else:
            raise Exception(f"Invalid dataset name: {dataset_cfg.name}; expected 'musiccaps' or 'audiocaps'")

        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        caption_idx = int(filename[:-4].split("/")[-1])
        caption = self.descriptions[caption_idx]

        return self._read_audio_as_mono(filename), caption

    def _read_audio_as_mono(self, filename: str) -> T:
        try:
            audio, sample_rate = audio_read(filename)
            print(f"{sample_rate=}")
        except Exception:
            print(f"Error reading audio file {filename}")
            raise

        if audio.shape[0] > 1:
            channels = audio.shape[0]
            audio = torch.unsqueeze(torch.sum(audio, dim=0) / channels, 0)

        print(f"{audio.shape[1]=}")

        return audio


def collate_fn(batch):
    # can't torch.stack as the audios have different sizes, using batch size of 1 only for now
    # @TODO: add proper batching, with returning padding mask to be used by tokenizer
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])
