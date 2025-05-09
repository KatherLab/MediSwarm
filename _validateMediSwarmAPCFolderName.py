#!/usr/bin/env python3

import sys

MediSwarmAPCFolderNameAllowList = {'minimal_training_pytorch_cnn', '3dcnn_ptl'}

if sys.argv[1] in MediSwarmAPCFolderNameAllowList:
    exit(0)
else:
    exit(1)
