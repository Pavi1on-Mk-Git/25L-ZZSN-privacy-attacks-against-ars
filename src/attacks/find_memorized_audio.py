from src.attacks import FeatureExtractor
from src.models.audiocraft import AudiocraftModelWrapper
from src.evaluation.audio_embedding import AudioEmbeddingModel
from torch import Tensor as T
import torch

import numpy as np
from torch import Tensor

from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity
import json


class ExtractMemorizedAudio(FeatureExtractor):
    top_tokens = {
        "audiogen_medium": [0, 1, 5, 14, 30],
    }

    # distance = {
    #     "var_30": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    #     "rar_xxl": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    #     "mar_h": lambda target, pred: torch.sqrt(torch.pow(target - pred, 2).sum(dim=1)).cpu(),
    # }

    # @torch.no_grad()
    # def get_features(self, img: T) -> T:
    #     model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to(self.model_cfg.device).eval()

    #     normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #     )
    #     skew_320 = transforms.Compose(
    #         [
    #             transforms.Resize([320, 320]),
    #             normalize,
    #         ]
    #     )
    #     features = model(skew_320(img))
    #     return features

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

        for pred, target, sample_index in tqdm(zip(preds, targets, sample_indices), total=len(sample_indices)):
            print(f"{pred.shape=}")
            N, K, T = pred.shape
            assert target.shape == (K, T)
            # continue

            pred = pred.to(device)
            target = target.to(device)

            pred_audios = self.model.tokens_to_audio(pred)
            target_audio = self.model.tokens_to_audio(target.unsqueeze(0))

            print(f"{pred_audios.shape=}")
            print(f"{target_audio.shape=}")

            pred_features = self.embedding_model.get_embeddings(pred_audios)
            target_features = self.embedding_model.get_embeddings(target_audio)

            print(f"{pred_features.shape=}")
            print(f"{target_features.shape=}")

            cosines = self.get_cosine(target_features, pred_features).cpu()
            print(f"{cosines.shape=}")

            return

        #     cosines = torch.cat(
        #         [
        #             torch.stack(
        #                 [self.get_cosine(features_real[[i]], features_pred[[i]]).cpu() for i in range(B)],
        #                 dim=0,
        #             )
        #             for features_pred in features_generated
        #         ],
        #         dim=1,
        #     ).T
        #     out.append(
        #         [
        #             batch[:, 5, -1].cpu(),
        #             batch[:, 5, -2].cpu(),
        #             *[self.distance[self.model_cfg.name](target, pred) for pred in preds],
        #             *cosines.tolist(),
        #         ]
        #     )

        # out = np.concatenate(out, axis=1).T
        # print(out.shape)
        # TOP_TOKENS = self.top_tokens[self.model_cfg.name]

        # df = pd.DataFrame(
        #     out,
        #     columns=[
        #         "sample_idx",
        #         "label",
        #         *[f"token_eq_{i}" for i in TOP_TOKENS],
        #         *[f"cosine_{i}" for i in TOP_TOKENS],
        #     ],
        # )
        # df.to_csv(
        #     f"analysis/plots/memorization/{self.model_cfg.name}_memorized_"
        #     f"{self.model_cfg.name}_{self.attack_cfg.std}.csv",
        #     index=False,
        # )
