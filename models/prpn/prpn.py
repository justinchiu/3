import torch
import torch.nn
import torch.nn.functional as F

from ..lm import LM


class Prpn(LM):
    def __init__(
        self,
        v_size,
        lut_dim,
        hid_dim,
        n_layers, # ?
        n_slots, # ?
        n_lookback, # kernel_w
        resolution, # ?
        res, # ?
        tie_weights=True,
        clip=5,
        dropout=0.4, # they have additional dropout options
    ):
        super(Prpn, self).__init__()

        self.lut = nn.Embedding(v_size, lut_dim)
        self.parser = None
        self.reader = None
        self.predicter = None
        self.decoder = None

        self.clip = clip
        self.rnn_parameters = []
