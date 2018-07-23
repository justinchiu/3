#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -m models.prpn.main_LM --cuda --data .data/penn-treebank/ptb. > prpn_ref.log
