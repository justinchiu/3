import argparse

from torchtext import data
from torchtext import datasets

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument()
    return args.get_args()

TEXT = data.Field(lower=True, batch_first=True)
TEXT.build_vocab(train)

train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=3, bptt_len=30, device="cuda:0")

train_iter, valid_iter, test_iter = datasets.WikiText2.iters(
    batch_size=4, bptt_len=30)

