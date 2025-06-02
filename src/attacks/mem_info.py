from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
from src.models import AudiocraftModelWrapper
import json
import os.path


class MemInfoExtractor(FeatureExtractor):
    def get_token_probs(self, tokens: T, probas: T) -> T:
        if len(probas.shape) == 3:
            return torch.gather(probas, 2, tokens.unsqueeze(2).long()).permute(0, 2, 1)  # B, 1, N_tokens
        else:
            return torch.gather(probas, -1, tokens.unsqueeze(-1).long()).permute(0, 3, 1, 2)  # B, 1, K, T

    def get_token_ranks(self, tokens: T, probas: T) -> T:
        top_k_indices = torch.topk(probas, probas.shape[-1], dim=-1).indices
        ranks = (top_k_indices == tokens.unsqueeze(-1)).long().argmax(dim=-1)
        return ranks.unsqueeze(1).float()  # (B, 1, N_tokens) or (B, 1, K, T)

    def get_token_losses(self, tokens: T, probas: T) -> T:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        if len(tokens.shape) == 3:
            return loss_fn(probas.permute(0, 2, 1), tokens).unsqueeze(1)  # B, 1, N_tokens
        else:
            return loss_fn(probas.permute(0, 3, 1, 2), tokens).unsqueeze(1)  # B, 1, K, T

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, conditions = batch
        B = images.shape[0]
        images = images.to(self.device)
        if type(conditions) is T:
            conditions = conditions.to(self.device).long()

        tokens = self.model.tokenize(images)
        logits = self.model.forward(images, conditions, is_cfg=False)

        if isinstance(self.model, AudiocraftModelWrapper):
            logits, mask = logits
            # logits, tokens = self.model.get_only_first_codebook(logits, tokens, mask)

        probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, K, T, card) or (B, N_tokens, card)

        token_probs = self.get_token_probs(tokens, probs)
        ranks = self.get_token_ranks(tokens, probs)
        is_top = ranks == 0
        max_probs = probs.max(dim=-1).values.unsqueeze(1)
        tokens_pred = logits.argmax(dim=-1).unsqueeze(1)
        features = torch.cat(
            [tokens.unsqueeze(1), is_top, ranks, max_probs, token_probs, tokens_pred],
            dim=1,
        )  # (B, N_features, N_tokens) or (B, N_features, K, T)

        if isinstance(self.model, AudiocraftModelWrapper):
            if os.path.isfile(self.captions_path_out):
                with open(self.captions_path_out, "r") as fp:
                    all_conditions = json.load(fp)
            else:
                all_conditions = []

            all_conditions.extend(conditions)

            with open(self.captions_path_out, "w") as fp:
                json.dump(all_conditions, fp)
        else:
            features = torch.cat([features, conditions.reshape(B, 1, 1).repeat(1, features.shape[1], 1)], dim=2)

        return features.cpu()
