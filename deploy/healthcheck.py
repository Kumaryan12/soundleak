"""Container health probe; does not instantiate an analysis session."""
import os
from urllib.request import urlopen

with urlopen('http://127.0.0.1:'+os.environ.get('PORT', '8501')+'/_stcore/health', timeout=4) as response:
    if response.status != 200:
        raise SystemExit(1)
