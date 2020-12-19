from encoder import EncoderTopped
from dataset import WordDataset
import numpy as np
import glob
import torch
import random
from torch import nn

if __name__ == "__main__":

    data_regex = "data/Airplane.txt"
    vocab_file = "assets/airplane_wordspace.vc"

    train_files = glob.glob(data_regex)
    words = WordDataset(random.sample(train_files, len(train_files)), batch_size=10, leadup_size=24, tokenize_method="WORDSPACE", vocab_file=vocab_file)
    model = EncoderTopped(len(words.vocab), 5, 8, 128, 64)
    model.load_state_dict(torch.load("checkpoint/model36999.pth"))

    start = words.encode("An airplane or aeroplane (informally plane) is a powered, fixed-wing aircraft")
    temp = 1

    print(len(start))

    intermed = start
    output = start.tolist()
    for ii in range(20):
        # print(intermed)
        pred = torch.from_numpy(intermed).unsqueeze(0)

        pred = model(pred)[0].detach().numpy()
        pred = np.exp(pred / temp)
        pred /= pred.sum()
        nextword = np.random.choice(np.arange(pred.shape[0]), p=pred)
        # nextword = np.argmax(pred)
        output += [nextword]

        if len(intermed) >= 12:
            intermed = intermed[len(intermed) - 11:]
        intermed = np.append(intermed, [nextword])
    # print(intermed)

    print(words.decode(output), "END")