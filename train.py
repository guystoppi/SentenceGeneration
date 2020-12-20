from model import Model
from dataset import WordDataset

import sys
import yaml
import os
import shutil
import numpy as np
import random
import glob

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def eval_model(model, dataset, inplst, device):
    model.eval()
    model.reset()

    total = []
    output = " ".join([dataset.vocab[idx] for idx in inplst]) + " | "

    for _ in range(30):
        inpt = torch.from_numpy(np.array(inplst)).unsqueeze(1).to(device)
        pred = torch.nn.functional.softmax(model(inpt)[0])
        num_vocab = pred.shape[0]
        pred_idx = np.argmax(pred.detach().cpu().numpy())
        inplst = inplst[1:] + [pred_idx]
        total += [pred_idx]

    output += " ".join([dataset.vocab[idx] for idx in total])

    model.train()
    model.reset()
    return output

if __name__ == "__main__":

    with open(sys.argv[1]) as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.Loader)

    words = WordDataset(**config["dataset"])
    device = config["device"]
    model = Model(**config["model"], vocab_size=len(words.vocab)).to(device)

    cross_entropy = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, **config["train"]["lr_decay"])

    writer = SummaryWriter(log_dir=config["logdir"])
    shutil.copy(sys.argv[1], config["logdir"])

    for ii in range(config["train"]["max_iter"]):
        query, label, file_skip = words.get_next_encoded()
        model.step()
        if file_skip:
            model.reset()

        query = query.astype(np.int64)
        label = label.astype(np.int64)

        query_torch = torch.from_numpy(query).to(device)
        label_torch = torch.from_numpy(label).to(device)

        pred = model(query_torch)

        acc = (np.argmax(pred.detach().cpu().numpy(), axis=1) == label).mean()

        loss_output = cross_entropy(pred, label_torch)
        loss_output.backward()
        optim.step()

        if ii in config["train"].get("lr_decay_steps", []):
            scheduler.step()

        writer.add_scalar("Loss", loss_output.item(), ii)
        writer.add_scalar("Accuracy", acc, ii)

        if (ii + 1) % config["train"]["eval_every"] == 0:        
            writer.add_text("Data Rotations", " ".join([str(val) for val in dataset.subpower_set]), ii)
            
            rand_idx = np.random.randint(query.shape[0])
            sentence_out = eval_model(model, words, query[rand_idx].tolist(), device)
            writer.add_text("Eval", sentence_out, ii)

        if (ii + 1) % config["train"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(config["logdir"], "model%05d.pth" % (ii)))
