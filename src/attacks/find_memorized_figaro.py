from src.attacks import FeatureExtractor
from src.models import FigaroWrapper
import torch
import numpy as np
from torch import Tensor as T
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from pretty_midi import PrettyMIDI, Note
import Levenshtein
import json


class ExtractMemorizedFigaro(FeatureExtractor):
    top_tokens = {
        "figaro": [1, 5, 14, 30],
    }

    distance = {
        "figaro": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    }

    def get_levenshtein(self, seq1: list[int], seq2: list[int]):
        return torch.tensor([1 - (Levenshtein.distance(seq1, seq2) / max(len(seq1), len(seq2)))])

    @torch.no_grad()
    def load_candidates(self) -> DataLoader:
        data = np.load(
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_memorized_{self.dataset_cfg.split}.npz",
            allow_pickle=True,
        )["data"]

        data = torch.from_numpy(data)
        data = data.reshape(data.shape[0], 5, -1)

        with open(
            f"{self.config.path_to_features}/{self.model_cfg.name}_mem_info_"
            + f"memorized_{self.dataset_cfg.split}_indexes.json",
            "r",
        ) as fh:
            indices = json.load(fh)

        indices = torch.tensor(indices)

        dataset = TensorDataset(indices, data)
        loader = DataLoader(
            dataset,
            num_workers=self.config.dataloader_num_workers,
            batch_size=self.model_cfg.batch_size,
            shuffle=False,
        )
        return loader

    def get_features(self, pms: list[PrettyMIDI]) -> list[int]:
        features = []

        for pm in pms:
            # Step 1: Gather all notes from all instruments
            all_notes = []
            for instrument in pm.instruments:
                for note in instrument.notes:
                    all_notes.append(note)

            # Step 2: Sort notes by pitch (descending), then by start time (ascending)
            sorted_notes = sorted(all_notes, key=lambda n: (-n.pitch, n.start))

            melody_notes = []

            def is_covered(note, melody_notes):
                """Return True if note's entire duration is covered by any melody note of equal or higher pitch"""
                for mel_note in melody_notes:
                    if mel_note.start <= note.start and mel_note.end >= note.end:
                        return True
                return False

            def split_note(note, melody_notes):
                """Split note into uncovered segments"""
                uncovered_segments = []
                current_start = note.start
                current_end = note.end
                for mel_note in melody_notes:
                    # Skip notes that don't overlap
                    if mel_note.end <= current_start or mel_note.start >= current_end:
                        continue
                    # Before overlap
                    if mel_note.start > current_start:
                        uncovered_segments.append((current_start, mel_note.start))
                    current_start = max(current_start, mel_note.end)
                if current_start < current_end:
                    uncovered_segments.append((current_start, current_end))
                return [Note(velocity=note.velocity, pitch=note.pitch, start=s, end=e) for s, e in uncovered_segments]

            # Step 3: Melody Extraction Loop
            while sorted_notes:
                note = sorted_notes.pop(0)
                if not is_covered(note, melody_notes):
                    split_notes = split_note(note, melody_notes)
                    melody_notes.extend(split_notes)

            melody_notes.sort(key=lambda n: n.start)

            features.append([note.pitch for note in melody_notes])

        return features

    def run(self, *args, **kwargs) -> None:
        self.model: FigaroWrapper

        loader = self.load_candidates()

        device = self.model_cfg.device
        out = []
        for sample_idx, batch in tqdm(loader, total=len(loader)):
            B = batch.shape[0]
            batch = batch.to(device).long()
            target = batch[:, 4, :]
            preds = [batch[:, idx, :] for idx in range(4)]

            out_preds = [self.model.tokens_to_img(pred) for pred in preds]
            out_target = self.model.tokens_to_img(target)
            features_real = self.get_features(out_target)
            features_generated = [self.get_features(pred) for pred in out_preds]
            levenshteins = torch.cat(
                [
                    torch.stack(
                        [self.get_levenshtein(features_real[i], features_pred[i]) for i in range(B)],
                        dim=0,
                    )
                    for features_pred in features_generated
                ],
                dim=1,
            ).T
            out.append(
                [
                    sample_idx,
                    *[self.distance[self.model_cfg.name](target, pred) for pred in preds],
                    *levenshteins.tolist(),
                ]
            )

        out = np.concatenate(out, axis=1).T
        print(out.shape)
        TOP_TOKENS = self.top_tokens[self.model_cfg.name]

        df = pd.DataFrame(
            out,
            columns=[
                "sample_idx",
                *[f"token_eq_{i}" for i in TOP_TOKENS],
                *[f"levenshtein_{i}" for i in TOP_TOKENS],
            ],
        )
        df.to_csv(
            f"analysis/plots/memorization/{self.model_cfg.name}_memorized_{self.dataset_cfg.split}.csv",
            index=False,
        )
