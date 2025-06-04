import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from figaro.datasets import MidiDataset, SeqCollator
from pathlib import Path
from figaro.models.vae import VqVaeModule
from transformers.models.bert.modeling_bert import BertAttention


def get_figaro_dataloader(config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig):
    data_dir = Path(dataset_cfg.data_dir)
    match dataset_cfg.split:
        case "train":
            split_files_data = Path(dataset_cfg.train_data)
        case "test":
            split_files_data = Path(dataset_cfg.test_data)
        case _:
            raise Exception(f"Invalid dataset split: {dataset_cfg.split}; expected 'train' or 'test'")

    with split_files_data.open("r") as f:
        split_paths = [line.strip() for line in f if line.strip()]

    files = []
    for rel_path in split_paths:
        file_path = data_dir / rel_path
        files.append(str(file_path))

    # assuming transformers>=4.36.0
    pl_ckpt = torch.load(model_cfg.vae_ckpt, weights_only=False, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    if "flavor" in kwargs:
        del kwargs["flavor"]
    if "vae_run" in kwargs:
        del kwargs["vae_run"]
    model = VqVaeModule(**kwargs)
    state_dict = pl_ckpt["state_dict"]
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError:
        # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
        bert_config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(
                bert_config, position_embedding_type=bert_config.position_embedding_type
            )
        model.load_state_dict(state_dict)
    model.freeze()
    model.eval()
    model.to(model_cfg.device)

    dataset = MidiDataset(
        midi_files=files,
        max_len=256,
        max_bars=256,
        description_flavor="latent",
        vae_module=model,
        device=model_cfg.device,
    )
    coll = SeqCollator(context_size=-1)

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=0,
        batch_size=model_cfg.batch_size,
        generator=g,
        collate_fn=coll,
    )
