"""
Adapted from https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
"""

import sys
import os

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")
sys.path.append("./audiocraft")

from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import torch
from src.local_datasets.audio_dataset import audio_read, convert_audio


SAMPLE_RATE = 16_000


class AudioEmbeddingModel:
    def __init__(self):
        self.device = torch.device("cuda")

        # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
        self.model = torch.hub.load(repo_or_dir="harritaylor/torchvggish", model="vggish")
        self.model.device = self.device

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, x):
        embd_lst = []

        for audio in x:
            audio = audio.squeeze(0)
            print(f"{audio.shape=}")
            embd = self.model.forward(audio, SAMPLE_RATE)
            print(f"{embd.shape=}")
            embd = embd.cpu()

            if torch.is_tensor(embd):
                embd = embd.detach().numpy()

            embd_lst.append(embd)

        return np.stack(embd_lst, axis=0).reshape((len(embd_lst), -1))


if __name__ == "__main__":

    def read(filename):
        audio, sample_rate = audio_read(filename)
        return convert_audio(audio, sample_rate, SAMPLE_RATE, 1)

    a1 = read("data/musiccaps/val/0.wav")
    a2 = read("data/musiccaps/val/1.wav")
    a3 = read("data/musiccaps/val/2.wav")

    print(f"{a1.shape=}")
    print(f"{a2.shape=}")
    print(f"{a3.shape=}")

    model = AudioEmbeddingModel()
    res = torch.tensor(model.get_embeddings([a1.numpy(), a2.numpy(), a3.numpy()]))

    print(f"{res.shape=}")

    sim = pairwise_cosine_similarity(res, res)
    print(sim)
