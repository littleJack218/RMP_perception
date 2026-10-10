#!/usr/bin/env python
"""Entry point: reproduce the paper's Fig. 4 metrics from the packaged
test split and checkpoints.  Run from the repository root:

    python run_inference.py
"""
from src.inference import main

if __name__ == '__main__':
    main()
