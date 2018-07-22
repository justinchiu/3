#!/bin/bash

python -m models.prpn.main_LM --cuda --data .data/penn-treebank/ptb. > prpn_ref.log
