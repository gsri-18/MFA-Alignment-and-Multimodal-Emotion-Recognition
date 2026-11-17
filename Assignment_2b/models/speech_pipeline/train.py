""" Trains the speech pipeline and keeps the best checkpoint by validation macro F1.

    python models/speech_pipeline/train.py --help

The blocks are in models/common/nets.py and the train loop in models/common/engine.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.common.engine import train

if __name__ == "__main__":
    train("speech")
