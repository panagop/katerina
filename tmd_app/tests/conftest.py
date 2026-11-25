import sys
from pathlib import Path

# Το core.py και οι σελίδες εισάγουν το `core` ως module του φακέλου tmd_app
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

# Η βάση αποτελεσμάτων σε προσωρινό φάκελο: οι έλεγχοι δεν γράφουν ποτέ στο results_db/
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ["TMD_RESULTS_DIR"] = tempfile.mkdtemp(prefix="tmd_results_test_")
