#!/bin/bash

echo HOSTNAME: $(hostname)

export PYTHONPATH=src:lib

python3 -m pdb lib/utils/main.py train_config.json
