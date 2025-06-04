from src.attacks import FeatureExtractor
from src.models import FigaroWrapper
from torch import Tensor as T
import torch

from tqdm import tqdm

import numpy as np
import json


class GenerateCandidatesFigaro(FeatureExtractor):
    top_tokens = {
        "figaro": [1, 5, 14, 30],
    }

    def get_data(self, split: str) -> tuple[np.ndarray, list[str]]:
        assert split == "train"
        features_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_train.npz"
        captions_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_train_latents.npy"
        bar_ids_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_train_bar_ids.npy"
        position_ids_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_train_position_ids.npy"

        features = np.load(features_filename, allow_pickle=True)["data"]
        latents = np.load(captions_filename)
        bar_ids = np.load(bar_ids_filename)
        position_ids = np.load(position_ids_filename)

        return features, latents, bar_ids, position_ids

    @torch.no_grad()
    def run(self, *args, **kwargs) -> T:
        self.model: FigaroWrapper

        TOP_TOKENS = self.top_tokens[self.model_cfg.name]
        members_features, latents, bar_ids, position_ids = self.get_data(self.dataset_cfg.split)
        members_features = torch.from_numpy(members_features)  # B, F, T
        bar_ids = torch.from_numpy(bar_ids)
        position_ids = torch.from_numpy(position_ids)
        latents = torch.from_numpy(latents)
        B, _, T = members_features.shape
        print("Data loaded")

        torch.manual_seed(0)
        scores = self.model.get_memorization_scores(members_features, 1)  # B
        assert scores.shape == (B,)

        ins = []

        for top_k in tqdm(range(self.attack_cfg.n_samples), desc="Getting Samples"):
            target_tokens, target_bar_ids, target_position_ids, sample_latent, sample_index = (
                self.model.get_target_label_memorization(
                    members_features, scores, latents, bar_ids, position_ids, top_k
                )
            )
            assert target_tokens.shape == (1, T)
            assert sample_index.shape == (1,)

            ins.append((target_tokens, target_bar_ids, target_position_ids, sample_latent, sample_index))

        out = []
        sample_indexes = []
        for target_tokens, target_bar_ids, target_position_ids, sample_latent, sample_index in tqdm(
            ins, desc="Generating Samples"
        ):
            pred = []
            for top_tokens in TOP_TOKENS:
                pred_tokens = self.model.generate_single_memorization(
                    top_tokens, target_tokens, target_bar_ids, target_position_ids, sample_latent
                )

                pred.append(pred_tokens)
                sample_indexes.append(sample_index.item())

            pred = torch.stack(pred, dim=1)
            out.append(torch.cat([pred, target_tokens.unsqueeze(1)], dim=1).cpu())

        out = torch.cat(out, dim=0).cpu().numpy()
        np.savez(
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            f"memorized_figaro_{self.dataset_cfg.split}.npz",
            data=out,
        )

        indices_filename = (
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            + f"memorized_figaro_{self.dataset_cfg.split}_indexes.json"
        )

        with open(indices_filename, "w") as fh:
            json.dump(sample_indexes, fh)
