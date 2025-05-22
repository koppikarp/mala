import os
import glob
import random
from typing import List, Iterator, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pytorch_lightning as pl
from transformers import AutoTokenizer
from Bio import SeqIO

__all__ = ["FastaIterableDataset", "BatchedFastaDataset", "FastaDataModule"]


class FastaIterableDataset(IterableDataset):
    """Streams individual sequences from FASTA, splits train/val at record-level."""

    def __init__(
        self,
        fasta_paths: List[str],
        tokenizer: AutoTokenizer,
        sample_fraction: float = 1.0,
        min_len: int = 20,
        val_fraction: float = 0.1,
        split: str = "train",
    ) -> None:
        super().__init__()
        if not fasta_paths:
            raise FileNotFoundError("fasta_paths list is empty")
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self._paths = fasta_paths
        self.tokenizer = tokenizer
        self.sample_fraction = sample_fraction
        self.min_len = min_len
        self.val_fraction = val_fraction
        self.split = split

    def _iter_files(self, files: List[str]) -> Iterator[Dict[str, Any]]:
        pad_id = self.tokenizer.pad_token_id
        for fp in files:
            with open(fp, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    if len(record.seq) < self.min_len:
                        continue
                    r = random.random()
                    # record-level train/val split
                    if self.split == "train":
                        if r < self.val_fraction:
                            continue
                        # optional subsample of training
                        if self.sample_fraction < 1.0 and random.random() > self.sample_fraction:
                            continue
                    else:  # val split
                        if r >= self.val_fraction:
                            continue
                    # tokenize
                    ids = self.tokenizer(
                        str(record.seq), add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].squeeze(0)
                    yield {
                        "input_ids": ids,
                        "attention_mask": (ids != pad_id).long(),
                        "seq_len": ids.size(0),
                        "seq_id": record.id,
                    }

    def __iter__(self):
        worker = get_worker_info()
        files = (
            self._paths
            if worker is None
            else self._paths[worker.id :: worker.num_workers]
        )
        return self._iter_files(files)


class BatchedFastaDataset(IterableDataset):
    """Wraps FastaIterableDataset to yield token-budgeted batches directly."""

    def __init__(self, base_ds: IterableDataset, max_tokens: int) -> None:
        super().__init__()
        self.base = base_ds
        self.max_tokens = max_tokens

    def __iter__(self):
        batch_ids: List[str] = []
        batch_tokens: List[torch.Tensor] = []
        token_cnt = 0
        pad_id = self.base.tokenizer.pad_token_id
        for sample in self.base:
            L = sample["seq_len"]
            if batch_tokens and token_cnt + L > self.max_tokens:
                yield self._make_batch(batch_ids, batch_tokens, pad_id)
                batch_ids, batch_tokens, token_cnt = [], [], 0
            batch_ids.append(sample["seq_id"])
            batch_tokens.append(sample["input_ids"])
            token_cnt += L
        if batch_tokens:
            yield self._make_batch(batch_ids, batch_tokens, pad_id)

    @staticmethod
    def _make_batch(ids: List[str], toks: List[torch.Tensor], pad_id: int) -> Dict[str, Any]:
        max_len = max(t.size(0) for t in toks)
        padded = torch.stack(
            [F.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in toks]
        )
        attention = (padded != pad_id).long()
        return {
            "input_ids": padded,
            "attention_mask": attention,
            "seq_id": ids,
            "seq_len": torch.tensor([t.size(0) for t in toks]),
        }


class FastaDataModule(pl.LightningDataModule):
    """Lightning DataModule with record-level train/val split and dynamic batching."""

    def __init__(
        self,
        fasta_dir: str = "fasta",
        tokenizer_name: str = "facebook/esm2_t33_650M_UR50D",
        max_tokens_per_batch: int = 2048,
        num_workers: int = 4,
        sample_fraction: float = 1.0,
        min_len: int = 20,
        val_fraction: float = 0.1,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer: Optional[AutoTokenizer] = None

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name
            )
        # gather all FASTA files
        all_files = sorted(
            glob.glob(
                os.path.join(self.hparams.fasta_dir, "**/*.fasta"),
                recursive=True,
            )
        )
        if self.hparams.shuffle:
            random.shuffle(all_files)

        # create train/val record-split datasets
        train_base = FastaIterableDataset(
            all_files,
            self.tokenizer,
            sample_fraction=self.hparams.sample_fraction,
            min_len=self.hparams.min_len,
            val_fraction=self.hparams.val_fraction,
            split="train",
        )
        val_base = FastaIterableDataset(
            all_files,
            self.tokenizer,
            sample_fraction=1.0,
            min_len=self.hparams.min_len,
            val_fraction=self.hparams.val_fraction,
            split="val",
        )

        self.train_dataset = BatchedFastaDataset(
            train_base, self.hparams.max_tokens_per_batch
        )
        self.val_dataset = BatchedFastaDataset(
            val_base, self.hparams.max_tokens_per_batch
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

