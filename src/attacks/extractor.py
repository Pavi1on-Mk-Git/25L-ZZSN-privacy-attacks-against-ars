import torch
from torch import Tensor as T
from typing import Tuple
from tqdm import tqdm
from src.attacks import DataSource
from src.models import GeneralVARWrapper, FigaroWrapper
from src.dataloaders import loaders
import os.path


class FeatureExtractor(DataSource):
    model: GeneralVARWrapper

    def process_batch(self, batch: Tuple[T, T], *args, **kwargs) -> T: ...

    @staticmethod
    def _is_all_same_length(features: list[T]) -> tuple[bool, int]:
        result = True

        B, F, T = features[0].shape
        max_length = T
        for batch in features:
            assert len(batch.shape) == 3
            assert (batch.shape[0], batch.shape[1]) == (B, F)
            if batch.shape[2] != T:
                result = False
            if batch.shape[2] > max_length:
                max_length = batch.shape[2]

        return result, max_length

    @staticmethod
    def _pad_with_nans(features: list[T], max_length: int) -> list[T]:
        result = []

        B, F, _ = features[0].shape
        for batch in features:
            if batch.shape[2] == max_length:
                continue

            padding = (
                torch.ones((B, F, max_length - batch.shape[2]), dtype=batch.dtype, device=batch.device) * torch.nan
            )
            batch = torch.concat([batch, padding], dim=2)
            assert batch.shape == (B, F, max_length)
            result.append(batch)

        return result

    def process_data(self, *args, **kwargs) -> T:
        assert self.model is not None
        loader = loaders[self.model_cfg.dataloader](self.config, self.model_cfg, self.dataset_cfg)
        features = []

        samples_processed = 0
        for batch in tqdm(loader, total=int(self.total_samples / self.model_cfg.batch_size + 1)):
            features.append(self.process_batch(batch, *args, **kwargs))
            if isinstance(self.model, FigaroWrapper):
                samples_processed += batch["input_ids"].shape[0]
            else:
                samples_processed += batch[0].shape[0]
            if samples_processed >= self.total_samples:
                break

        all_same_length, max_length = self._is_all_same_length(features)
        if not all_same_length:
            features = self._pad_with_nans(features, max_length)

        return torch.cat(features, dim=0)[: self.total_samples]

    def check_data(self, data: T) -> None:
        """
        Check that the data is in the correct format
        """
        assert len(data.shape) >= 3  # N_samples, N_measurements, *Features
        # assert data.shape[0] == self.total_samples

    def run(self, *args, **kwargs) -> None:
        """
        Run the feature extractor
        """
        if os.path.isfile(self.captions_path_out):
            os.remove(self.captions_path_out)

        # 1. Collect features for members and nonmembers
        features = self.process_data(*args, **kwargs)
        # 2. Run assertions
        self.check_data(features)
        # 3. Save features
        self.save(features)
