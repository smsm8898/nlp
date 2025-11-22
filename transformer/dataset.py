import torch
import lightning as L
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path):
        self.bpe = spm.SentencePieceProcessor()
        self.bpe.load(model_path) # .model file

    def __len__(self):
        return self.bpe.vocab_size()

    def encode(self, sentence: str):
        return self.bpe.encode_as_ids(sentence)
    
    def decode(self, indices, skip_special_tokens=True):
        return self.bpe.decode_ids(indices)
    
    @property
    def pad_idx(self):
        return self.bpe.pad_id()
    
    @property
    def sos_idx(self):
        return self.bpe.bos_id()
    
    @property
    def eos_idx(self):
        return self.bpe.eos_id()

    @property
    def unk_idx(self):
        return self.bpe.unk_id()


class WMT14Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            src_file:str, 
            tgt_file:str, 
            src_tokenizer:Tokenizer, 
            tgt_tokenizer:Tokenizer, 
            max_len:int,
        ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f]

        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f]
        

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # Encode
        src_indices = self.src_tokenizer.encode(self.src_sentences[idx])
        tgt_indices = self.tgt_tokenizer.encode(self.tgt_sentences[idx])

        # Truncate if too long
        src_indices = src_indices[: self.max_len]
        tgt_indices = tgt_indices[: self.max_len-2] # sos, eos token

        # Add SOS(start of sentence) and EOS(end of sentence) to Target
        tgt_indices = [self.tgt_tokenizer.sos_idx] + tgt_indices + [self.tgt_tokenizer.eos_idx]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
    

class WMT14DataModule(L.LightningDataModule):
    def __init__(
            self,
            train_src_file: str,
            train_tgt_file: str,
            val_src_file: str,
            val_tgt_file: str,
            src_tokenizer_file: str,
            tgt_tokenizer_file: str,
            batch_size: int,
            num_workers: int,
            max_len: int,
        ):
        super().__init__()
        self.train_src_file = train_src_file
        self.train_tgt_file = train_tgt_file
        self.val_src_file = val_src_file
        self.val_tgt_file = val_tgt_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len

        # Tokenizer
        self.src_tokenizer = Tokenizer(src_tokenizer_file)
        self.tgt_tokenizer = Tokenizer(tgt_tokenizer_file)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Dataset
            self.train_dataset = WMT14Dataset(
                self.train_src_file, 
                self.train_tgt_file, 
                self.src_tokenizer, 
                self.tgt_tokenizer, 
                self.max_len
            )
            self.val_dataset = WMT14Dataset(
                self.val_src_file, 
                self.val_tgt_file, 
                self.src_tokenizer, 
                self.tgt_tokenizer, 
                self.max_len
            )

    def _collate_fn(self, batch):
        src_batch, tgt_batch = [], []

        for src, tgt in batch:
            src_batch.append(src)
            tgt_batch.append(tgt)

        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_batch,
            batch_first=True, 
            padding_value=self.src_tokenizer.pad_idx
        )
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, 
            batch_first=True, 
            padding_value=self.tgt_tokenizer.pad_idx
        )
        return src_batch, tgt_batch

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers, # MPS 최적화: 0
            pin_memory=False, # MPS 불필요
            persistent_workers=False, # MPS 자동(False)
            collate_fn=self._collate_fn,
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=self._collate_fn,
        )
