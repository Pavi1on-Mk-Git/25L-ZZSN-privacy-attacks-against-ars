from src.local_datasets.dataset import ImageFolderDataset
from src.local_datasets.audio_dataset import AudioDataset

datasets = {
    "imagenet": ImageFolderDataset,
    "musiccaps": AudioDataset,
}
