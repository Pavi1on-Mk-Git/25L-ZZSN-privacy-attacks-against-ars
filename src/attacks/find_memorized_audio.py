from src.attacks import FeatureExtractor
from src.models.audiocraft import AudiocraftModelWrapper
from src.evaluation.audio_embedding import AudioEmbeddingModel
from torch import Tensor as T
import torch

import numpy as np
import pandas as pd
from torch import Tensor

from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity
import json


class ExtractMemorizedAudio(FeatureExtractor):
    top_tokens = {
        "audiogen_medium": [0, 1, 5, 14, 30],
    }

    @staticmethod
    def distance(target, pred):
        return (target == pred).sum(dim=1).cpu()

    def get_cosine(self, features_real: T, features_generated: T) -> T:
        cosine_similarity = (
            pairwise_cosine_similarity(features_real, features_generated).median(0).values.detach().cpu()
        )
        return cosine_similarity

    @torch.no_grad()
    def load_candidates(self) -> tuple[Tensor, Tensor, list[int], list[str]]:
        data = np.load(
            f"out/features/{self.model_cfg.name}_mem_info_memorized_audiocaps_{self.dataset_cfg.split}.npz",
            allow_pickle=True,
        )["data"]

        data = torch.from_numpy(data)
        pred, target = data[:, :5], data[:, 5]

        indices_filename = (
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            + f"memorized_audiocaps_{self.dataset_cfg.split}_indexes.json"
        )

        with open(indices_filename, "r") as fh:
            sample_indices = json.load(fh)

        return pred, target, sample_indices

    def run(self, *args, **kwargs) -> None:
        self.model: AudiocraftModelWrapper
        self.embedding_model = AudioEmbeddingModel()

        preds, targets, sample_indices = self.load_candidates()
        device = self.model_cfg.device

        out = []

        for pred, target, sample_index in tqdm(zip(preds, targets, sample_indices), total=len(sample_indices)):
            print(f"{pred.shape=}")
            N, K, T = pred.shape
            assert target.shape == (K, T)

            pred = pred.to(device)
            target = target.to(device)

            pred_audios = self.model.tokens_to_audio(pred)
            target_audio = self.model.tokens_to_audio(target.unsqueeze(0))

            pred_features = self.embedding_model.get_embeddings(pred_audios)
            target_features = self.embedding_model.get_embeddings(target_audio)

            assert pred_features.shape[0] == N
            N, F = pred_features.shape
            assert target_features.shape == (1, F)

            cosines = self.get_cosine(target_features, pred_features).cpu()
            assert cosines.shape == (N,)

            out.append(
                [
                    sample_index,
                    *[self.distance(target, single_pred) for single_pred in pred],
                    *cosines.tolist(),
                ]
            )

        out = np.concatenate(out, axis=1).T
        print(out.shape)
        TOP_TOKENS = self.top_tokens[self.model_cfg.name]

        df = pd.DataFrame(
            out,
            columns=[
                "sample_idx",
                *[f"token_eq_{i}" for i in TOP_TOKENS],
                *[f"cosine_{i}" for i in TOP_TOKENS],
            ],
        )
        df.to_csv(
            f"analysis/plots/memorization/{self.model_cfg.name}_memorized_{self.model_cfg.name}.csv",
            index=False,
        )
