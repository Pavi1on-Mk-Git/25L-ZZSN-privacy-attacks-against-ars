from src.models.GeneralVARWrapper import GeneralVARWrapper
import torch
from torch import Tensor as T
from audiocraft.models import MusicGen


class AudiocraftModelWrapper(GeneralVARWrapper):
    def load_models(self):
        """
        Loads the generator and tokenizer models
        """
        match self.model_cfg.name:
            case "musicgen_small":
                model = MusicGen.get_pretrained("small")
                return model.lm, model.compression_model
            case _:
                raise Exception(f"unknown model name: {self.model_cfg.name}")

    def tokenize(self, images: T) -> T:
        """
        Tokenizes the images, return tensor of shape (batch_size, seq_len)
        """
        raise NotImplementedError

    def forward(self, images: T, conditioning: T) -> T:
        """
        Computes logits of all tokens, returns tensor of shape (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError

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
    def get_loss_per_token(self, images: T, classes: T, *args, **kwargs) -> T:
        """
        Computes the loss per token, returns tensor of shape (batch_size, seq_len)
        """
        tokens = self.tokenize(images)
        logits = self.forward(images, classes)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(logits.permute(0, 2, 1), tokens)  # B, N_tokens

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
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        raise NotImplementedError
