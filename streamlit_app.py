"""Community Cloud entrypoint for audio and recorded Wi-Fi evaluation labs."""
import os
from pathlib import Path
import runpy

os.environ['SOUNDLEAK_HOSTED'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
runpy.run_path(str(Path(__file__).with_name('app.py')), run_name='__main__')
