#!/bin/bash

set -e

echo "Starting to convert the official BEVFusion checkpoint to the current architecture..."
python convert_ckpt.py
echo "Conversion completed!"
