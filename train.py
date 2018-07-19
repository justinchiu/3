import argparse
import random
import math

from torchtext import data
from torchtext import datasets

import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler


from data import StructuredLmDataset

from models import Prpn
from models.args import Args

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--devid", type=int, default=-1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--bsz", type=int, default=64)
    args.add_argument("--bptt", type=int, default=35)
    args.add_argument("--dropout", type=float, default=0.7)
    args.add_argument("--idropout", type=float, default=0.5)
    args.add_argument("--rdropout", type=float, default=0.5)
    args.add_argument("--clip", type=float, default=1.)

    args.add_argument("--lut_dim", type=int, default=800)
    args.add_argument("--hid_dim", type=int, default=1200)
    args.add_argument("--n_layers", type=int, default=2)
    args.add_argument("--n_slots", type=int, default=5)
    args.add_argument("--resolution", type=int, default=0.1)
    args.add_argument("--res", type=int, default=0)
    args.add_argument("--nlookback", type=int, default=5)

    args.add_argument("--report_interval", type=int, default=25)
    args.add_argument("--seed", type=int, default=1234)

    return args.parse_args()
args = get_args()


device = torch.device("cpu" if args.devid < 0 else "cuda:{}".format(args.devid))
random.seed(args.seed)
torch.manual_seed(args.seed)
if "cuda" in device.type:
    torch.cuda.manual_seed(args.seed)

TEXT = data.Field(lower=True, batch_first=False)
#TEXT = data.Field(lower=True, batch_first=True)
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
    (train, valid, test),
    batch_size=args.bsz,
    bptt_len=args.bptt,
    device=device,
    repeat=False,
)

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
model.to(device)
print(model)

optimizer = optim.Adam(
    #[p for p in model.parameters() if p.requires_grad],
    model.parameters(),
    lr = 3e-3,
    #betas = (0.9, 0.999),
    betas = (0., 0.999),
    eps = 1e-9,
    weight_decay = 1e-6,
)

schedule = scheduler = lr_scheduler.ReduceLROnPlateau(
    #optimizer, 'min', 0.5, patience=0, threshold=0)
    optimizer, 'min', 0.5, patience=2, threshold=0)

train_args = Args(
    optimizer = optimizer,
    clip = args.clip,
    report_interval = args.report_interval,
)

for e in range(args.epochs):
    train_loss, train_ntokens = model.train_epoch(train_iter, train_args)
    loss, ntokens = model.validate(valid_iter)
    print("Train loss: {}; ppl: {}"
        .format(train_loss / train_ntokens, math.exp(train_loss / train_ntokens)))
    print("Valid loss: {}; ppl: {}"
        .format(loss / ntokens, math.exp(loss / ntokens)))
    schedule.step(loss)
