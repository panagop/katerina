import sys
from pathlib import Path

# Τα modules του tmd_parametric εισάγονται ως απλά modules (όπως στα scripts του φακέλου)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
