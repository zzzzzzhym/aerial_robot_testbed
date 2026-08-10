import sys
import os

subpackage_path = os.path.abspath(os.path.dirname(__file__))

if subpackage_path not in sys.path:
    sys.path.insert(0, subpackage_path)
