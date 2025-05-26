import os
import os.path
from torch.utils.data import Dataset
import torch
from torch.types import Tensor as T
import pandas as pd
from audiocraft.data.audio import audio_read
from audiocraft.data.audio_utils import convert_audio

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")


# @TODO: perhaps add partitioning per gpu like in the image dataset, though number of GPUs is set to 1 anyway
class AudioDataset(Dataset):
    def __init__(self, dataset_cfg):
        self.sample_rate = dataset_cfg.sample_rate
        self.channels = dataset_cfg.channels

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
        print(f"caption={self.descriptions[caption_idx]}")
        caption = preprocess_caption(self.descriptions[caption_idx])

        print(f"preprocessed={caption}")

        return self._read_audio_as_mono(filename), caption

    def _read_audio_as_mono(self, filename: str) -> T:
        try:
            audio, sample_rate = audio_read(filename)
            audio = convert_audio(audio, sample_rate, self.sample_rate, self.channels)
        except Exception:
            print(f"Error reading audio file {filename}")
            raise

        return audio


def preprocess_caption(caption: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(caption)
    tagged = pos_tag(tokens)

    tagged = [
        (word, tag)
        for word, tag in tagged
        if not word.lower() in stop_words and word not in string.punctuation and not word.isdigit()
    ]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, _get_wordnet_sentence_position(tag)) for word, tag in tagged]

    return " ".join(tokens)


def _get_wordnet_sentence_position(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def collate_fn(batch):
    # can't torch.stack as the audios have different sizes, using batch size of 1 only for now
    # @TODO: add proper batching, with returning padding mask to be used by tokenizer
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])
