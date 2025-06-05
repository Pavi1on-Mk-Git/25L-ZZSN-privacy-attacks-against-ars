from src.models.GeneralVARWrapper import GeneralVARWrapper
import torch
from torch import Tensor as T
from figaro.models.seq2seq import Seq2SeqModule
from figaro.constants import PAD_TOKEN
from figaro.input_representation import remi2midi
from transformers.models.bert.modeling_bert import BertAttention
from pretty_midi import PrettyMIDI


def load_from_checkpoint(model_type, checkpoint):
    # assuming transformers>=4.36.0
    pl_ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    if "flavor" in kwargs:
        del kwargs["flavor"]
    if "vae_run" in kwargs:
        del kwargs["vae_run"]
    model = model_type(**kwargs)
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
    return model


class FigaroWrapper(GeneralVARWrapper):

    def load_models(self):
        """
        Loads the generator and tokenizer models
        """
        model = load_from_checkpoint(Seq2SeqModule, self.model_cfg.latent_ckpt)
        model.to(self.model_cfg.device)

        return model, None

    def tokenize(self, batch: dict[str, T]) -> T:
        """
        Tokenizes the images, return tensor of shape (batch_size, seq_len)
        """
        return batch["input_ids"]

    def forward(self, batch: dict[str, T], is_cfg: bool = False) -> T:
        """
        Computes logits of all tokens, returns tensor of shape (batch_size, seq_len, vocab_size)
        """
        if not is_cfg:
            return self.generator(
                x=batch["input_ids"][:, : self.generator.context_size],
                z=batch["latents"],
                bar_ids=batch["bar_ids"][:, : self.generator.context_size],
                position_ids=batch["position_ids"][:, : self.generator.context_size],
                description_bar_ids=None,
            )
        else:
            self.generator.description_flavor = "none"
            out = self.generator(
                x=batch["input_ids"][:, : self.generator.context_size],
                z=None,
                bar_ids=batch["bar_ids"][:, : self.generator.context_size],
                position_ids=batch["position_ids"][:, : self.generator.context_size],
                description_bar_ids=None,
            )
            self.generator.description_flavor = "latent"
            return out

    @torch.no_grad()
    def get_token_list(self, images: T, *args, **kwargs) -> list[T]:
        """
        Returns list of tokens, each tensor of shape (batch_size, n_tokens -- may vary)
        """
        raise NotImplementedError

    def tokens_to_token_list(self, tokens: T) -> list[T]:
        raise NotImplementedError

    def get_memorization_scores(self, members_features: T, ft_idx: int) -> T:
        return members_features[:, ft_idx, -100:-1].mean(dim=1)

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
    def generate_single_memorization(
        self, top: int, target_tokens: T, target_bar_ids: T, target_position_ids: T, latent: T
    ) -> T:
        B, T = target_tokens.shape
        assert B == 1

        batch = {
            "input_ids": target_tokens[:, :top],
            "bar_ids": target_bar_ids[:, :top],
            "position_ids": target_position_ids[:, :top],
            "latents": latent,
        }

        pad_index = torch.argmax(torch.any((target_tokens == 0), dim=1).int())

        if pad_index == 0:
            pad_index = T

        generated = self.generator.sample(batch=batch, max_length=pad_index - 1, temp=0.0)

        generated_padding = torch.zeros(
            (generated["sequences"].shape[0], T - generated["sequences"].shape[1]),
            dtype=generated["sequences"].dtype,
            device=generated["sequences"].device,
        )

        return torch.cat([generated["sequences"], generated_padding], dim=1)

    @torch.no_grad()
    def tokens_to_img(self, tokens: T, *args, **kwargs) -> list[PrettyMIDI]:
        midis = []
        for i in range(tokens.shape[0]):
            events = self.generator.vocab.decode(tokens[i].cpu())
            midi = remi2midi(events)
            midis.append(midi)
        return midis

    def get_target_label_memorization(
        self, tokens: T, scores: T, latents: T, bar_ids: T, position_ids: T, k: int
    ) -> tuple[T, T, T, T, T]:
        mem_samples_indices = torch.topk(scores, len(scores)).indices
        sample_index = mem_samples_indices[k]

        pad_index = torch.argmax(torch.any((latents[sample_index] == 0), dim=1).int())

        if pad_index != 0:
            target_latents = latents[[sample_index], :pad_index, :]
        else:
            target_latents = latents[[sample_index], :, :]

        return (
            tokens[[sample_index], :].to(self.model_cfg.device).long(),
            bar_ids[[sample_index], :].to(self.model_cfg.device).long(),
            position_ids[[sample_index], :].to(self.model_cfg.device).long(),
            target_latents.to(self.model_cfg.device),
            sample_index.unsqueeze(0).to(self.model_cfg.device),
        )

    @torch.no_grad()
    def get_loss_for_tokens(self, preds: T, ground_truth: T) -> T:
        preds = preds.permute(0, 2, 1)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.generator.vocab.to_i(PAD_TOKEN), reduction="none")
        return loss_fn(preds, ground_truth)

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        raise NotImplementedError
