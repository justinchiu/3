import argparse

from torchtext import data
from torchtext import datasets
from data import StructuredLmDataset

from models import Prpn

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--devid", type=int, default=-1)
    args.add_argument("--bsz", type=int, default=64)
    args.add_argument("--bptt", type=int, default=50)
    args.add_argument("--dropout", type=float, default=0.7)
    args.add_argument("--idropout", type=float, default=0.5)
    args.add_argument("--rdropout", type=float, default=0.5)
    args.add_argument("--clip", type=float, default=1.)

    args.add_argument("--lut_dim", type=int, default=512)
    args.add_argument("--hid_dim", type=int, default=1024)
    args.add_argument("--n_layers", type=int, default=2)
    args.add_argument("--n_slots", type=int, default=5)
    args.add_argument("--resolution", type=int, default=0.1)
    args.add_argument("--res", type=int, default=0)
    args.add_argument("--nlookback", type=int, default=5)



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

#model = LM()
model = Prpn(
    ntoken = len(TEXT.vocab),
    ninp = args.lut_dim,
    nhid = args.hid_dim,
    nslots = args.n_slots,
    nlayers = args.n_layers,
    resolution = args.resolution,
    res = args.res,
    tie_weights = True,
    dropout = args.dropout,
    idropout = args.idropout,
    rdropout = args.rdropout,
)
