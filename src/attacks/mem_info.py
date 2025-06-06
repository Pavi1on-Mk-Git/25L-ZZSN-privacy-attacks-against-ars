from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
from src.models import AudiocraftModelWrapper, FigaroWrapper
import json
import os.path
import numpy as np


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
        if isinstance(self.model, FigaroWrapper):
            return self.process_batch_figaro(batch)

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

    def process_batch_figaro(self, batch: dict[str, T]):
        self.model: FigaroWrapper

        device_batch = {key: value.to(self.device) for key, value in batch.items() if key != "files"}

        tokens = self.model.tokenize(device_batch)
        logits = self.model.forward(device_batch)
        bar_ids = batch["bar_ids"]
        position_ids = batch["position_ids"]
        latents = batch["latents"]

        probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, K, T, card) or (B, N_tokens, card)

        token_probs = self.get_token_probs(tokens[:, : self.model.generator.context_size], probs)
        ranks = self.get_token_ranks(tokens[:, : self.model.generator.context_size], probs)
        is_top = ranks == 1
        max_probs = probs.max(dim=-1).values.unsqueeze(1)
        tokens_pred = logits.argmax(dim=-1).unsqueeze(1)
        features = torch.cat(
            [is_top, ranks, max_probs, token_probs, tokens_pred],
            dim=1,
        )  # (B, N_features, N_tokens) or (B, N_features, K, T)

        if os.path.isfile(self.tokens_path_out):
            all_tokens = torch.from_numpy(np.load(self.tokens_path_out))
            new_len = max(all_tokens.shape[1], tokens.shape[1])

            all_tokens_padding = torch.zeros(
                (all_tokens.shape[0], new_len - all_tokens.shape[1]),
                dtype=all_tokens.dtype,
                device=all_tokens.device,
            )
            tokens_padding = torch.zeros(
                (tokens.shape[0], new_len - tokens.shape[1]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            all_tokens = torch.cat([all_tokens, all_tokens_padding], dim=1)
            tokens = torch.cat([tokens, tokens_padding], dim=1)
            all_tokens = torch.cat([all_tokens, tokens.cpu()])
        else:
            all_tokens = tokens.cpu()

        np.save(self.tokens_path_out, all_tokens)

        if os.path.isfile(self.latents_path_out):
            all_latents = torch.from_numpy(np.load(self.latents_path_out))
            new_len = max(all_latents.shape[1], latents.shape[1])

            all_conditions_padding = torch.zeros(
                (all_latents.shape[0], new_len - all_latents.shape[1], all_latents.shape[2]),
                dtype=all_latents.dtype,
                device=all_latents.device,
            )
            latents_padding = torch.zeros(
                (latents.shape[0], new_len - latents.shape[1], latents.shape[2]),
                dtype=latents.dtype,
                device=latents.device,
            )
            all_latents = torch.cat([all_latents, all_conditions_padding], dim=1)
            latents = torch.cat([latents, latents_padding], dim=1)
            all_latents = torch.cat([all_latents, latents])
        else:
            all_latents = latents

        np.save(self.latents_path_out, all_latents)

        if os.path.isfile(self.bar_ids_path_out):
            all_bar_ids = torch.from_numpy(np.load(self.bar_ids_path_out))
            new_len = max(all_bar_ids.shape[1], bar_ids.shape[1])

            all_conditions_padding = torch.zeros(
                (all_bar_ids.shape[0], new_len - all_bar_ids.shape[1]),
                dtype=all_bar_ids.dtype,
                device=all_bar_ids.device,
            )
            bar_ids_padding = torch.zeros(
                (bar_ids.shape[0], new_len - bar_ids.shape[1]),
                dtype=bar_ids.dtype,
                device=bar_ids.device,
            )
            all_bar_ids = torch.cat([all_bar_ids, all_conditions_padding], dim=1)
            bar_ids = torch.cat([bar_ids, bar_ids_padding], dim=1)
            all_bar_ids = torch.cat([all_bar_ids, bar_ids])
        else:
            all_bar_ids = bar_ids

        np.save(self.bar_ids_path_out, all_bar_ids)

        if os.path.isfile(self.position_ids_path_out):
            all_position_ids = torch.from_numpy(np.load(self.position_ids_path_out))
            new_len = max(all_position_ids.shape[1], position_ids.shape[1])

            all_conditions_padding = torch.zeros(
                (all_position_ids.shape[0], new_len - all_position_ids.shape[1]),
                dtype=all_position_ids.dtype,
                device=all_position_ids.device,
            )
            position_ids_padding = torch.zeros(
                (
                    position_ids.shape[0],
                    new_len - position_ids.shape[1],
                ),
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
            all_position_ids = torch.cat([all_position_ids, all_conditions_padding], dim=1)
            position_ids = torch.cat([position_ids, position_ids_padding], dim=1)
            all_position_ids = torch.cat([all_position_ids, position_ids])
        else:
            all_position_ids = position_ids

        np.save(self.position_ids_path_out, all_position_ids)

        return features.cpu()
