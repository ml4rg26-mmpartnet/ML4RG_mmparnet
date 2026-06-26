#!/usr/bin/env python
"""CLI wrapper for the protein+cell FiLM multitask training workflow."""
from __future__ import annotations

import os

from mmpartnet.experiments.film_multitask import train_main


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    train_main()
