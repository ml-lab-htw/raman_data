import subprocess
import sys


def test_import_raman_data_does_not_import_zenodo_get():
    code = "import sys, raman_data; assert 'zenodo_get' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
