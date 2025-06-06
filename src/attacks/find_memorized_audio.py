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

from audiocraft.data.audio import audio_write
from pathlib import Path
import shutil


PADDING_TOKEN = -9223372036854775808


class ExtractMemorizedAudio(FeatureExtractor):
    top_tokens = {
        "audiogen_medium": [0, 1, 5, 14, 30],
    }

    def get_mem_info_data(self, split: str) -> tuple[np.ndarray, list[str]]:
        filename_base = f"out/features/{self.model_cfg.name}_mem_info_10k_audiocaps_{split}"
        features_filename = f"{filename_base}.npz"
        captions_filename = f"{filename_base}_conditions.json"

        features = np.load(features_filename, allow_pickle=True)["data"]
        with open(captions_filename, "r") as fh:
            captions = json.load(fh)

        return features, captions

    @staticmethod
    def distance(target: Tensor, pred: Tensor):
        return (target == pred).sum().cpu()

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
        csv_dir = Path("analysis/plots/memorization")
        csv_dir.mkdir(parents=True, exist_ok=True)

        samples_dir = Path("generated_samples")
        shutil.rmtree(samples_dir)
        samples_dir.mkdir(parents=True)

        mem_info_features, captions = self.get_mem_info_data(self.dataset_cfg.split)
        scores = self.model.get_memorization_scores(mem_info_features, 1)

        TOP_TOKENS = self.top_tokens[self.model_cfg.name]

        self.model: AudiocraftModelWrapper
        self.embedding_model = AudioEmbeddingModel()

        preds, targets, sample_indices = self.load_candidates()
        device = self.model_cfg.device

        out = []

        for pred, target, sample_index in tqdm(zip(preds, targets, sample_indices), total=len(sample_indices)):
            N, K, T = pred.shape
            assert target.shape == (K, T)

            pred = pred.to(device)
            target = target.to(device)

            first_pad_index = (target == PADDING_TOKEN).any(dim=0).to(torch.int8).argmax()
            if first_pad_index > 0:
                target = target[:, :first_pad_index]
                pred = pred[:, :, :first_pad_index]

            pred_audios = self.model.tokens_to_audio(pred)
            target_audio = self.model.tokens_to_audio(target.unsqueeze(0))

            for pred_audio, prefix_size in zip(pred_audios, TOP_TOKENS):
                audio_write(f"{samples_dir}/{sample_index}_{prefix_size}", pred_audio, 16_000)
            audio_write(f"{samples_dir}/{sample_index}_target", target_audio.squeeze(0), 16_000)

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
                    scores[sample_index],
                    captions[sample_index],
                    *[self.distance(target, single_pred) for single_pred in pred],
                    *cosines.tolist(),
                ]
            )

        df = pd.DataFrame(
            out,
            columns=[
                "sample_idx",
                "caption",
                *[f"token_eq_{i}" for i in TOP_TOKENS],
                *[f"cosine_{i}" for i in TOP_TOKENS],
            ],
        )
        df.sort_values(by=f"cosine_{TOP_TOKENS[-1]}", ascending=False, inplace=True)
        df.to_csv(
            csv_dir / f"{self.model_cfg.name}_memorized_{self.model_cfg.name}_{self.dataset_cfg.split}.csv",
            index=False,
        )
