import numpy as np
import os
import logging
import random

def make_even_split(datalst, num_entries):

    while len(datalst) < num_entries:
        sizelst = np.array([x.shape[0] for x in datalst])

        max_size, sub_size = np.max(sizelst), np.mean(sizelst).astype(np.int32)

        if max_size == 1:
            return datalst
        if max_size - sub_size == 0:
            sub_size = max_size // 2

        max_set = datalst.pop(np.argmax(sizelst))
        datalst += [max_set[:sub_size], max_set[sub_size:]]

    return datalst


class WordDataset:

    def __init__(self, files, batch_size=25, leadup_size=12, vocab_file=None, dataset_sizes=None, num_rotations=0, rotation_frequency=1000):
        
        self.batch_step = 0
        self.data_idx = 0
        self.batch_size = batch_size
        self.leadup_size = leadup_size

        self.vocab_file = vocab_file if vocab_file else "assets/vocab.vc"
        self.files = random.sample(files, len(files))

        self.num_rotations = num_rotations
        self.rotation_frequency = rotation_frequency        
        self.rotation_dataset = []

        self.build_vocab()
        self.dataset = self.load_all_files()

        self.dataset_sizes = dataset_sizes
        
    def encode(self, sentence):

        idxs = []
        for tkn in sentence:
            idxs.append(self.vocab.index(tkn))
    
        return np.array(idxs, dtype=np.int64)

    def decode(self, idxs):
        return [self.vocab[idx] for idx in idxs]

    def load_all_files(self):
        
        if self.num_rotations == 0:
            dataset = np.concatenate([self.load_file(idx) for idx in range(len(self.files))])
            dataset = dataset.reshape(self.batch_size, -1)
        else:
            # num rotations will be equal to max(self.num_rotations, self.num_files)
            file_datasets = [self.load_file(j) for j in range(len(self.files))]
            for j in range(len(file_datasets)):
                datalen = self.batch_size * (file_datasets[j].shape[0] // self.batch_size)
                file_datasets[j] = file_datasets[j][:datalen].reshape(self.batch_size, -1).transpose(1,0)
            dataset = [data.transpose(1,0) for data in make_even_split(file_datasets, self.num_rotations)]

        return dataset

    def load_file(self, file_idx):
        with open(self.files[file_idx]) as fl: 
            file_text = fl.read().split("\n")
            trunc = self.batch_size * (len(file_text) // self.batch_size) - 1
            file_text = file_text[:trunc] + ["__pageend"]

            dataset = np.array(self.encode(file_text))

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
        elif type(self.dataset_sizes[0]) == list:
            for datasize, trainstep in self.dataset_sizes:
                if self.batch_step < trainstep:
                    self.data_limit = datasize
                    break
        else:
            self.data_limit = self.dataset_sizes[0] * (self.batch_step // self.dataset_sizes[1] + 1)

    def get_next_data_rotation(self):
        if self.batch_step % self.rotation_frequency == 0:
            self.subpower_set = np.random.randint(2, size=self.num_rotations)
            if self.subpower_set.sum() == 0:
                self.subpower_set[np.random.randint(self.num_rotations)] = 1

            self.rotation_dataset = np.concatenate([
                self.dataset[j] for j in range(len(self.dataset)) if self.subpower_set[j]
            ], axis=1)


    def get_next_encoded(self):

        if self.dataset_sizes:
            self.get_current_data_limit()

            self.data_idx %= self.data_limit
            file_skip = self.data_idx == 0
            dataset = self.dataset
        elif self.num_rotations > 0:
            self.get_next_data_rotation()
            dataset = self.rotation_dataset
            file_skip = self.batch_step % self.rotation_frequency == 0

            self.data_idx %= dataset.shape[1] // self.leadup_size - 1
        else:
            dataset = self.dataset
            file_skip = False

            self.data_idx %= dataset.shape[1] // self.leadup_size - 1


        query = dataset[:, self.data_idx * self.leadup_size:(self.data_idx + 1) * self.leadup_size]
        label = dataset[:, (self.data_idx + 1) * self.leadup_size]
        self.data_idx += 1
        self.batch_step += 1

        return query, label, file_skip