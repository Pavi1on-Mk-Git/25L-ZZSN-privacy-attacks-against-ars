from src.attacks import FeatureExtractor
from src.models import AudiocraftModelWrapper
from torch import Tensor as T
import torch

from tqdm import tqdm

import numpy as np
import json
from math import ceil


class GenerateCandidatesAudio(FeatureExtractor):
    top_tokens = {
        "audiogen_medium": [0, 1, 5, 14, 30],
    }

    def get_data(self, split: str) -> tuple[np.ndarray, list[str]]:
        assert split == "train"
        features_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_audiocaps_train.npz"
        captions_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_audiocaps_train_conditions.json"

        features = np.load(features_filename, allow_pickle=True)["data"]
        with open(captions_filename, "r") as fh:
            captions = json.load(fh)

        return features, captions

    def batched(self, iterable):
        n = self.attack_cfg.batch_size
        length = len(iterable)
        for index in range(0, length, n):
            yield iterable[index : min(index + n, length)]

    @torch.no_grad()
    def run(self, *args, **kwargs) -> T:
        self.model: AudiocraftModelWrapper

        TOP_TOKENS = self.top_tokens[self.model_cfg.name]
        members_features, captions = self.get_data(self.dataset_cfg.split)
        members_features = torch.from_numpy(members_features)  # B, F, T
        B, F, K, T = members_features.shape
        print("Data loaded")

        torch.manual_seed(0)
        scores = self.model.get_memorization_scores(members_features, 1)  # B
        assert scores.shape == (B,)

        ins = []

        for top_k in tqdm(range(self.attack_cfg.n_samples), desc="Getting Samples"):
            target_tokens, sample_caption, sample_index = self.model.get_target_label_memorization(
                members_features, scores, captions, top_k
            )
            assert target_tokens.shape == (1, K, T)
            assert sample_index.shape == (1,)

            ins.append((target_tokens, sample_caption, sample_index.item()))

        out = []
        sample_indexes = []
        for batch in tqdm(
            self.batched(ins), desc="Generating Samples", total=ceil(len(ins) / self.attack_cfg.batch_size)
        ):
            target_tokens = []
            batch_caption = []
            batch_indices = []

            for sample_in in batch:
                single_target_tokens, single_sample_caption, single_sample_index = sample_in
                target_tokens.append(single_target_tokens)
                batch_caption.append(single_sample_caption)
                batch_indices.append(single_sample_index)

            target_tokens = torch.concat(target_tokens, dim=0)

            pred = []
            for top_tokens in TOP_TOKENS:
                pred_tokens = self.model.generate_single_memorization(top_tokens, target_tokens, batch_caption)

                pred.append(pred_tokens)

            pred = torch.stack(pred, dim=1)
            out.append(torch.cat([pred, target_tokens.unsqueeze(1)], dim=1).cpu())
            sample_indexes.extend(batch_indices)

        out = torch.cat(out, dim=0).cpu().numpy()
        np.savez(
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            f"memorized_audiocaps_{self.dataset_cfg.split}.npz",
            data=out,
        )

        indices_filename = (
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            + f"memorized_audiocaps_{self.dataset_cfg.split}_indexes.json"
        )

        with open(indices_filename, "w") as fh:
            json.dump(sample_indexes, fh)
