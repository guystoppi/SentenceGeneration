import numpy as np
import os
import logging
import random

class WordDataset:

    def __init__(self, files, batch_size=25, leadup_size=12, vocab_file=None, dataset_sizes=None):
        
        self.batch_step = 0
        self.data_idx = 0
        self.batch_size = batch_size
        self.leadup_size = leadup_size

        self.vocab_file = vocab_file if vocab_file else "assets/vocab.vc"
        self.files = random.sample(files, len(files))

        self.build_vocab()
        self.dataset = self.load_all_files()

        self.dataset_sizes = np.array(dataset_sizes)

    def encode(self, sentence):

        idxs = []
        for tkn in sentence:
            idxs.append(self.vocab.index(tkn))
    
        return np.array(idxs, dtype=np.int64)

    def decode(self, idxs):
        return [self.vocab[idx] for idx in idxs]


    def load_all_files(self):
        
        dataset = np.concatenate([self.load_file(idx) for idx in range(len(self.files))])
        dataset = dataset.reshape(self.batch_size, -1)

        return dataset

    def load_file(self, file_idx):
        with open(self.files[file_idx]) as fl: 
            file_text = fl.read().split("\n")
            trunc = self.batch_size * (len(file_text) // self.batch_size) - 1
            file_text = file_text[:trunc] + ["__pageend"]

            dataset = np.array(self.encode(file_text))

        dataset = dataset # N
        return dataset

    def build_vocab(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file) as vocab_file:
                self.vocab = vocab_file.read().split("\n")
        else:

            logging.info("Pre-existing vocab file not found, generating new one...")
            self.vocab = set()
            for j, file in enumerate(self.files):
                logging.info("%d / %d files have been read", j, len(self.files))
                with open(file) as f:
                    file_text = f.read() + "\n__pageend"
                    self.vocab.update(file_text.split("\n"))
            logging.info("All files have been read, saving vocab file for future reference...")

            self.vocab = sorted(list(self.vocab))

            os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
            with open(self.vocab_file, "w") as vocab_file:
                vocab_file.write("\n".join(list(self.vocab)))

    def get_current_data_limit(self):
        if self.dataset_sizes is None:
            self.data_limit = self.dataset.shape[1] // self.leadup_size
        elif len(self.dataset_sizes.shape) > 1:
            for datasize, trainstep in self.dataset_sizes:
                if self.batch_step < trainstep:
                    self.data_limit = datasize
                    break
        else:
            self.data_limit = self.dataset_sizes[0] * (self.batch_step // self.dataset_sizes[1] + 1)

    def get_next_encoded(self):

        self.get_current_data_limit()

        self.data_idx %= self.data_limit
        file_skip = self.data_idx == 0

        query = self.dataset[:, self.data_idx:self.data_idx + self.leadup_size]
        label = self.dataset[:, self.data_idx + self.leadup_size]
        self.data_idx += 1
        self.batch_step += 1

        return query, label, file_skip