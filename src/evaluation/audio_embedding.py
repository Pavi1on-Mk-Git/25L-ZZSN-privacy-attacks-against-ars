"""
Adapted from https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
"""

import numpy as np
import torch


SAMPLE_RATE = 16_000


class AudioEmbeddingModel:
    def __init__(self):
        self.device = torch.device("cuda")

        # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
        self.model = torch.hub.load(repo_or_dir="harritaylor/torchvggish", model="vggish")
        self.model.device = self.device

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, x: torch.Tensor):
        embd_lst = []

        for audio in x:
            audio = audio.squeeze(0).numpy()
            embd = self.model.forward(audio, SAMPLE_RATE)
            embd = embd.cpu()

            if torch.is_tensor(embd):
                embd = embd.detach().numpy()

            embd_lst.append(embd)

        return np.stack(embd_lst, axis=0).reshape((len(embd_lst), -1))
