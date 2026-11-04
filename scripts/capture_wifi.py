"""Collect this Mac's Wi-Fi link measurements without network identifiers."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or not shutil.which('swift'):
        parser.error('This collector needs macOS and Apple Swift command-line tools.')
    if not 5 <= args.duration <= 300 or not .2 <= args.interval <= 2:
        parser.error('Use duration 5–300 seconds and interval 0.2–2 seconds.')
    if args.output.exists():
        parser.error('Output exists. Choose a new file to preserve earlier captures.')
    print('Keep the Mac and router stationary. Keep the first 10 seconds as a static reference.', flush=True)
    script = Path(__file__).with_suffix('.swift')
    try:
        result = subprocess.run(['swift', str(script), '--duration', str(args.duration),
                                 '--interval', str(args.interval)], capture_output=True,
                                timeout=args.duration + 90)
    except subprocess.TimeoutExpired:
        parser.exit(1, 'Collector timed out; no capture saved.\n')
    if result.returncode:
        sys.stderr.buffer.write(result.stderr)
        parser.exit(1, 'Capture failed; no capture saved.\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        output.write(result.stdout)
    print('Saved RSSI measurements:', args.output)
    print('Upload this CSV in Soundleak → Wi-Fi lab. It does not establish human presence.')


if __name__ == '__main__':
    main()
