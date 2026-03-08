import sys
import os

# Get the absolute path of the subpackage directory
subpackage_path = os.path.abspath(os.path.dirname(__file__))

# Add the subpackage directory to sys.path if it's not already there
if subpackage_path not in sys.path:
    sys.path.insert(0, subpackage_path)
