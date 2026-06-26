#!/usr/bin/env python
"""CLI wrapper for standalone FiLM multitask checkpoint evaluation."""
from __future__ import annotations

import os

from mmpartnet.experiments.film_multitask import eval_main


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mmpartnet_pycache")
    eval_main()
