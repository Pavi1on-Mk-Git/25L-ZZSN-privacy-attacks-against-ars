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

    def get_data(self) -> tuple[np.ndarray, list[str]]:
        features_filename = f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_{self.dataset_cfg.split}.npz"
        tokens_filename = (
            f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_{self.dataset_cfg.split}_tokens.npy"
        )
        captions_filename = (
            f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_{self.dataset_cfg.split}_latents.npy"
        )
        bar_ids_filename = (
            f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_{self.dataset_cfg.split}_bar_ids.npy"
        )
        position_ids_filename = (
            f"out/features/{self.model_cfg.name}_mem_info_10k_lakhmidi_{self.dataset_cfg.split}_position_ids.npy"
        )

        features = np.load(features_filename, allow_pickle=True)["data"]
        tokens = np.load(tokens_filename)
        latents = np.load(captions_filename)
        bar_ids = np.load(bar_ids_filename)
        position_ids = np.load(position_ids_filename)

        return features, tokens, latents, bar_ids, position_ids

    @torch.no_grad()
    def run(self, *args, **kwargs) -> T:
        self.model: FigaroWrapper

        TOP_TOKENS = self.top_tokens[self.model_cfg.name]
        members_features, tokens, latents, bar_ids, position_ids = self.get_data()
        members_features = torch.from_numpy(members_features)  # B, F, T
        tokens = torch.from_numpy(tokens)
        bar_ids = torch.from_numpy(bar_ids)
        position_ids = torch.from_numpy(position_ids)
        latents = torch.from_numpy(latents)
        B, _, T = members_features.shape
        print("Data loaded")

        torch.manual_seed(0)
        scores = self.model.get_memorization_scores(members_features, 0)  # B
        assert scores.shape == (B,)

        ins = []

        for top_k in tqdm(range(self.attack_cfg.n_samples), desc="Getting Samples"):
            target_tokens, target_bar_ids, target_position_ids, sample_latent, sample_index = (
                self.model.get_target_label_memorization(tokens, scores, latents, bar_ids, position_ids, top_k)
            )
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
                ).cpu()

                pred.append(pred_tokens)
                self.model.tokens_to_img(pred_tokens)[0].write(
                    f"pred_{sample_index.item()}_{top_tokens}_{self.dataset_cfg.split}.mid"
                )

            sample_indexes.append(sample_index.item())
            pred = torch.stack(pred, dim=1)
            out.append(torch.cat([pred, target_tokens.unsqueeze(1).cpu()], dim=1))

            self.model.tokens_to_img(target_tokens)[0].write(
                f"target_{sample_index.item()}_{self.dataset_cfg.split}.mid"
            )

        out = torch.cat(out, dim=0).cpu().numpy()
        np.savez(
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_memorized_{self.dataset_cfg.split}.npz",
            data=out,
        )

        indices_filename = (
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            + f"memorized_{self.dataset_cfg.split}_indexes.json"
        )

        scores_sorted = torch.topk(scores, len(scores)).values.tolist()[: self.attack_cfg.n_samples]

        with open(indices_filename, "w") as fh:
            json.dump({"indexes": sample_indexes, "scores": scores_sorted}, fh)
