""" Evaluates the best fusion checkpoint on the test part of the split.

    python models/fusion_pipeline/test.py --help

The blocks are in models/common/nets.py and the test loop in models/common/engine.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.common.engine import test

if __name__ == "__main__":
    test("fusion")
