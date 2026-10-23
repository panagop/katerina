import sys
from pathlib import Path

# Το core.py και οι σελίδες εισάγουν το `core` ως module του φακέλου tmd_app
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
