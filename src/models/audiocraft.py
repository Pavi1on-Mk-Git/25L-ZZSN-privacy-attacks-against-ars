from src.models.GeneralVARWrapper import GeneralVARWrapper
import torch
from torch import Tensor as T
from audiocraft.models import MusicGen, AudioGen
from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft.solvers.musicgen import MusicGenSolver


class AudiocraftModelWrapper(GeneralVARWrapper):
    def is_audio_model(self) -> bool:
        return True

    def load_models(self):
        """
        Loads the generator and tokenizer models
        """
        match self.model_cfg.name:
            case "musicgen_small":
                model = MusicGen.get_pretrained("facebook/musicgen-small")
            case "audiogen_medium":
                model = AudioGen.get_pretrained("facebook/audiogen-medium")
            case _:
                raise Exception(f"unknown model name: {self.model_cfg.name}")

        self.autocast = model.autocast
        return model.lm, model.compression_model

    @torch.no_grad()
    def tokenize(self, audios: T) -> T:
        """
        Tokenizes the images, return tensor of shape (batch_size, seq_len)
        """
        tokens, _ = self.tokenizer.encode(audios)

        return tokens  # B, K, T

    @torch.no_grad()
    def forward(self, audios: T, conditioning: list[str], is_cfg: bool) -> tuple[T, T]:
        attributes = [ConditioningAttributes(text={"description": description}) for description in conditioning]

        tokenized = self.generator.condition_provider.tokenize(attributes)
        condition_tensors = self.generator.condition_provider(tokenized)

        tokens, _ = self.tokenizer.encode(audios)

        with self.autocast:
            out = self.generator.compute_predictions(tokens, [], condition_tensors)
        return (out.logits, out.mask)  # (B, K, T, card), (B, K, T)

    def flatten_with_mask(self, logits: T, tokens: T, mask: T) -> tuple[T, T]:
        B, K, T, card = logits.shape
        assert tokens.shape == (B, K, T)
        assert mask.shape == (B, K, T)
        assert B == 1  # assuming batch_size = 1

        logits = torch.concat([token_logits for token_logits in logits.squeeze(0).transpose(0, 1)], axis=0)
        tokens = torch.concat([token for token in tokens.squeeze(0).transpose(0, 1)], axis=0)
        mask = torch.concat([token_mask for token_mask in mask.squeeze(0).transpose(0, 1)], axis=0)

        assert logits.shape == (K * T, card)
        assert tokens.shape == (K * T,)
        assert mask.shape == (K * T,)

        logits = logits[mask].unsqueeze(0)
        tokens = tokens[mask].unsqueeze(0)

        # result_tokens = []
        # result_logits = []

        # for k in range(K):
        #     logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # T, card
        #     tokens_k = tokens[:, k, ...].contiguous().view(-1)  # T
        #     mask_k = mask[:, k, ...].contiguous().view(-1)  # T

        #     ce_logits = logits_k[mask_k]
        #     ce_tokens = tokens_k[mask_k]

        #     result_logits.append(ce_logits)
        #     result_tokens.append(ce_tokens)

        # logits = torch.concat(result_logits, dim=0).unsqueeze(0)  # 1, seq_len, card
        # tokens = torch.concat(result_tokens, dim=0).unsqueeze(0)  # 1, seq_len

        return logits, tokens

    @torch.no_grad()
    def get_token_list(self, images: T, *args, **kwargs) -> list[T]:
        """
        Returns list of tokens, each tensor of shape (batch_size, n_tokens -- may vary)
        """
        raise NotImplementedError

    def tokens_to_token_list(self, tokens: T) -> list[T]:
        raise NotImplementedError

    def get_memorization_scores(self, members_features: T, ft_idx: int) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def get_loss_per_token(self, audios: T, conditioning: T, *args, **kwargs) -> T:
        """
        Computes the loss per token, returns tensor of shape (batch_size, seq_len)
        """
        tokens = self.tokenize(audios)
        logits, mask = self._forward_with_mask(audios, conditioning)

        MusicGenSolver._compute_cross_entropy(None, logits, tokens, mask)

        # loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # return loss_fn(logits.permute(0, 2, 1), tokens)  # B, N_tokens
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        folder: str,
        n_samples_per_class: int = 10,
        std: float = 0,
        clamp_min: float = float("-inf"),
        clamp_max: float = float("inf"),
    ) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def generate_single_memorization(self, top: int, target_token_list: list[T], label: T, std: float) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def tokens_to_img(self, tokens: T, *args, **kwargs) -> T:
        raise NotImplementedError

    def get_target_label_memorization(
        self, members_features: T, scores: T, sample_classes: T, cls: int, k: int
    ) -> tuple[T, T, T]:
        raise NotImplementedError

    @torch.no_grad()
    def get_loss_for_tokens(self, preds: T, ground_truth: T) -> T:
        B, n_tokens, _ = preds.shape
        assert ground_truth.shape == (B, n_tokens)

        preds = preds.permute(0, 2, 1)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(preds, ground_truth)  # B, seq_len

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        raise NotImplementedError
