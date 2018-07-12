import argparse

from torchtext import data
from torchtext import datasets
from data import StructuredLmDataset

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("-devid", type=int, default=-1)
    args.add_argument("-bsz", type=int, default=1)
    args.add_argument("-bptt", type=int, default=50)
    return args.parse_args()
args = get_args()

device = "cpu" if args.devid < 0 else "cuda:{}".format(args.devid)

#TEXT = data.Field(lower=True, batch_first=False)
TEXT = data.Field(lower=True, batch_first=True)
"""
train, valid, test = StructuredLmDataset.splits(
    path="data/wikitext-2",
    train="wiki.train.tokens",
    validation="wiki.valid.tokens",
    test="wiki.test.tokens",
    text_field=TEXT)
"""
train, valid, test = datasets.PennTreebank.splits(text_field=TEXT)
TEXT.build_vocab(train)

train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, bptt_len=args.bptt, device=device)

