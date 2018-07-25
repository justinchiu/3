#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 python -m models.prpn.main_LM --cuda --data .data/penn-treebank/ptb. --tied --hard > prpn_ref_noshuffle.log
CUDA_VISIBLE_DEVICES=3 python -m models.prpn.main_LM --cuda --data .data/penn-treebank/ptb. --tied --hard > prpn_ref_shuffle.log
