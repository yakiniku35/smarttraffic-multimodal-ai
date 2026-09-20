"""pytest 設定。

沒有這個檔案的話，直接執行 `pytest` 不會把專案根目錄放進 sys.path，
tests/ 底下的測試就會出現 `ModuleNotFoundError: No module named 'config'`
（只有 `python -m pytest` 剛好會動）。放一個根目錄的 conftest.py
就能讓 README 裡寫的 `pytest -q` 正常運作。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
