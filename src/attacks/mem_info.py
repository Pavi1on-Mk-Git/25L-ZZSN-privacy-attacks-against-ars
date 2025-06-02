from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
from src.models import AudiocraftModelWrapper
import json


class MemInfoExtractor(FeatureExtractor):
    def get_token_probs(self, tokens: T, probas: T) -> T:
        return torch.gather(probas, 2, tokens.unsqueeze(2).long()).permute(0, 2, 1)  # B, 1, N_tokens

    def get_token_ranks(self, tokens: T, probas: T) -> T:
        top_k_indices = torch.topk(probas, probas.shape[2], dim=2).indices
        ranks = (top_k_indices == tokens.unsqueeze(2)).long().argmax(dim=2)
        return ranks.unsqueeze(1).float()  # B, 1, N_tokens

    def get_token_losses(self, tokens: T, probas: T) -> T:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(probas.permute(0, 2, 1), tokens).unsqueeze(1)  # B, 1, N_tokens

    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        images, conditions = batch
        images = images.to(self.device)
        if type(conditions) is T:
            conditions = conditions.to(self.device).long()

        tokens = self.model.tokenize(images)
        logits = self.model.forward(images, conditions, is_cfg=False)

        if isinstance(self.model, AudiocraftModelWrapper):
            logits, mask = logits
            logits, tokens = self.model.get_only_first_codebook(logits, tokens, mask)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        token_probs = self.get_token_probs(tokens, probs)
        ranks = self.get_token_ranks(tokens, probs)
        is_top = ranks == 0
        max_probs = probs.max(dim=2).values.unsqueeze(1)
        tokens_pred = logits.argmax(dim=2).unsqueeze(1)
        features = torch.cat(
            [tokens.unsqueeze(1), is_top, ranks, max_probs, token_probs, tokens_pred],
            dim=1,
        )  # B, N_features, N_tokens

        # features = torch.cat([features, classes.reshape(B, 1, 1).repeat(1, features.shape[1], 1)], dim=2)

        with open(self.path_out.replace(".npz", "_conditions.json"), "w+") as fp:
            old_conditions = json.load(fp)
            fp.seek(0)
            fp.truncate(0)

            old_conditions.extend(conditions)
            json.dump(old_conditions, fp)

        # print(features.shape)
        return features.cpu()
