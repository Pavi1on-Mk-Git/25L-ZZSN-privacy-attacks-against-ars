from src.attacks import FeatureExtractor
from src.models import AudiocraftModelWrapper
from torch import Tensor as T
import torch

from tqdm import tqdm

import numpy as np
import json


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

            ins.append((target_tokens, sample_caption, sample_index))

        out = []
        sample_indexes = []
        for target_tokens, sample_caption, sample_index in tqdm(ins, desc="Generating Samples"):
            pred = []
            for top_tokens in TOP_TOKENS:
                pred_tokens = self.model.generate_single_memorization(top_tokens, target_tokens, sample_caption)
                print(f"{pred_tokens.shape=}")

                pred.append(pred_tokens)
                sample_indexes.append(sample_index.item())

            pred = torch.stack(pred, dim=1)
            print(f"{pred.shape=}")
            print(f"{target_tokens.unsqueeze(1).shape=}")
            print(f"{torch.cat([pred, target_tokens.unsqueeze(1)], dim=1).shape=}")
            out.append(torch.cat([pred, target_tokens.unsqueeze(1)], dim=1).cpu())

        out = torch.cat(out, dim=0).cpu().numpy()
        print(f"{out.shape=}")
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
