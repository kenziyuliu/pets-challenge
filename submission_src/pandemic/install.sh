#!/bin/bash

# This script is run by the submission system as a startup script.
# For our submission we disabled the GPU since the final model is fairly simple.

# Disable GPU
export CUDA_VISIBLE_DEVICES=

