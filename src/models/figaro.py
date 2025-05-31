from src.models.GeneralVARWrapper import GeneralVARWrapper
import torch
from torch import Tensor as T
from figaro.models.seq2seq import Seq2SeqModule
from transformers.models.bert.modeling_bert import BertAttention


class FigaroWrapper(GeneralVARWrapper):

    def load_models(self):
        """
        Loads the generator and tokenizer models
        """
        # assuming transformers>=4.36.0
        pl_ckpt = torch.load(self.model_cfg.expert_ckpt, weights_only=False, map_location="cpu")
        kwargs = pl_ckpt["hyper_parameters"]
        if "flavor" in kwargs:
            del kwargs["flavor"]
        if "vae_run" in kwargs:
            del kwargs["vae_run"]
        model = Seq2SeqModule(**kwargs)
        state_dict = pl_ckpt["state_dict"]
        # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
        try:
            # succeeds for checkpoints trained with transformers>4.13.0
            model.load_state_dict(state_dict)
        except RuntimeError:
            # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
            config = model.transformer.decoder.bert.config
            for layer in model.transformer.decoder.bert.encoder.layer:
                layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
            model.load_state_dict(state_dict)
        model.freeze()
        model.eval()
        return model, None

    def tokenize(self, batch: dict[str, T]) -> T:
        """
        Tokenizes the images, return tensor of shape (batch_size, seq_len)
        """
        return batch["input_ids"]

    def forward(self, batch: dict[str, T]) -> T:
        """
        Computes logits of all tokens, returns tensor of shape (batch_size, seq_len, vocab_size)
        """
        return self.generator(
            x=batch["input_ids"],
            z=batch["description"],
            bar_ids=batch["bar_ids"],
            position_ids=batch["position_ids"],
            description_bar_ids=batch["desc_bar_ids"],
        )

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
    def get_loss_for_tokens(self, batch: dict[str, T]) -> T:
        return self.generator.get_loss(batch)

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        raise NotImplementedError
