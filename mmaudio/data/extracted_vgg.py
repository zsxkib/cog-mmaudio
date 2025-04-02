import logging
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset

from mmaudio.utils.dist_utils import local_rank

log = logging.getLogger()


class ExtractedVGG(Dataset):

    def __init__(
        self,
        tsv_path: Union[str, Path],
        *,
        premade_mmap_dir: Union[str, Path],
        data_dim: dict[str, int],
    ):
        super().__init__()

        self.data_dim = data_dim
        self.df_list = pd.read_csv(tsv_path, sep='\t').to_dict('records')
        self.ids = [d['id'] for d in self.df_list]

        log.info(f'Loading precomputed mmap from {premade_mmap_dir}')
        # load precomputed memory mapped tensors
        premade_mmap_dir = Path(premade_mmap_dir)
        td = TensorDict.load_memmap(premade_mmap_dir)
        log.info(f'Loaded precomputed mmap from {premade_mmap_dir}')
        self.mean = td['mean']
        self.std = td['std']
        self.clip_features = td['clip_features']
        self.sync_features = td['sync_features']
        self.text_features = td['text_features']

        if local_rank == 0:
            log.info(f'Loaded {len(self)} samples.')
            log.info(f'Loaded mean: {self.mean.shape}.')
            log.info(f'Loaded std: {self.std.shape}.')
            log.info(f'Loaded clip_features: {self.clip_features.shape}.')
            log.info(f'Loaded sync_features: {self.sync_features.shape}.')
            log.info(f'Loaded text_features: {self.text_features.shape}.')

        assert self.mean.shape[1] == self.data_dim['latent_seq_len'], \
            f'{self.mean.shape[1]} != {self.data_dim["latent_seq_len"]}'
        assert self.std.shape[1] == self.data_dim['latent_seq_len'], \
            f'{self.std.shape[1]} != {self.data_dim["latent_seq_len"]}'

        assert self.clip_features.shape[1] == self.data_dim['clip_seq_len'], \
            f'{self.clip_features.shape[1]} != {self.data_dim["clip_seq_len"]}'
        assert self.sync_features.shape[1] == self.data_dim['sync_seq_len'], \
            f'{self.sync_features.shape[1]} != {self.data_dim["sync_seq_len"]}'
        assert self.text_features.shape[1] == self.data_dim['text_seq_len'], \
            f'{self.text_features.shape[1]} != {self.data_dim["text_seq_len"]}'

        assert self.clip_features.shape[-1] == self.data_dim['clip_dim'], \
            f'{self.clip_features.shape[-1]} != {self.data_dim["clip_dim"]}'
        assert self.sync_features.shape[-1] == self.data_dim['sync_dim'], \
            f'{self.sync_features.shape[-1]} != {self.data_dim["sync_dim"]}'
        assert self.text_features.shape[-1] == self.data_dim['text_dim'], \
            f'{self.text_features.shape[-1]} != {self.data_dim["text_dim"]}'

        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.mean
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def get_memory_mapped_tensor(self) -> TensorDict:
        td = TensorDict({
            'mean': self.mean,
            'std': self.std,
            'clip_features': self.clip_features,
            'sync_features': self.sync_features,
            'text_features': self.text_features,
        })
        return td

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = {
            'id': self.df_list[idx]['id'],
            'a_mean': self.mean[idx],
            'a_std': self.std[idx],
            'clip_features': self.clip_features[idx],
            'sync_features': self.sync_features[idx],
            'text_features': self.text_features[idx],
            'caption': self.df_list[idx]['label'],
            'video_exist': self.video_exist,
            'text_exist': self.text_exist,
        }

        return data

    def __len__(self):
        return len(self.ids)
