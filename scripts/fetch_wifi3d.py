"""Fetch a small official MM-Fi subset and the pinned CSI2PointCloud checkpoint.

Only selected compressed members are downloaded from E01.zip. The archive's
central directory is read with HTTP ranges; the complete 22 GB archive is not
downloaded. Dataset: CC BY-NC 4.0, for noncommercial research use.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ID = '1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B'
ARCHIVE_SIZE = 22_301_071_119
CHECKPOINT_ID = '1YKkG6l9cMvq92pIne6MxQrLLdogY9WRQ'
CHECKPOINT_SIZE = 364_623_506
CHECKPOINT_SHA256 = '7a624013000a34f7f24a939e4b69e305bb4c6dd0d3bdc8623ed869170dc128cb'
UPSTREAM_COMMIT = 'a180c3b6b55d00d142166dc0735856d7c5772b55'
FRAME_IDS = (1, 51, 101, 151, 201, 251)


def session():
    value = requests.Session()
    value.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])))
    return value


def file_hash(path):
    value = sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(2 ** 20), b''):
            value.update(chunk)
    return value.hexdigest()


def range_bytes(client, start, count):
    if start < 0 or count <= 0 or count > 150_000_000 or start + count > ARCHIVE_SIZE:
        raise ValueError('Invalid or excessive archive range.')
    end = start + count - 1
    params = {'id': ARCHIVE_ID, 'export': 'download', 'confirm': 't',
              'soundleak_range': f'{start}-{end}'}
    with client.get('https://drive.usercontent.google.com/download', params=params,
                    headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=60) as response:
        response.raise_for_status()
        expected = f'bytes {start}-{end}/{ARCHIVE_SIZE}'
        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
            raise ValueError('Archive server did not honor the requested range; refusing a full download.')
        data = response.content
        if len(data) != count:
            raise ValueError('Truncated archive range.')
        return data


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.client = session()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else ARCHIVE_SIZE + offset
        return self.pos

    def read(self, count=-1):
        count = min(count if count >= 0 else ARCHIVE_SIZE - self.pos, ARCHIVE_SIZE - self.pos)
        if count <= 0:
            return b''
        data = range_bytes(self.client, self.pos, count)
        self.pos += count
        return data

    def close(self):
        self.client.close()
        super().close()


def subset_index():
    prefix = 'E01/S05/A05/'
    with RangeFile() as source, zipfile.ZipFile(source) as archive:
        return [{'name': p.filename, 'offset': p.header_offset, 'compressed': p.compress_size,
                 'size': p.file_size, 'crc': p.CRC, 'compression': p.compress_type}
                for p in archive.infolist() if p.filename.startswith(prefix)
                and p.filename.endswith(('.mat', '.bin'))]


def fetch_member(entry, root):
    # Names must belong to the fixed selected recording, not arbitrary archive paths.
    path = Path(entry['name'])
    if path.parts[:3] != ('E01', 'S05', 'A05') or '..' in path.parts:
        raise ValueError('Unexpected member path.')
    target = root/path
    if target.exists() and target.stat().st_size == entry['size']:
        data = target.read_bytes()
        if zlib.crc32(data) & 0xffffffff == entry['crc']:
            return {'path': str(path), 'bytes': len(data), 'sha256': sha256(data).hexdigest()}
    with session() as client:
        header = range_bytes(client, entry['offset'], 30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP member header.')
        name_length, extra_length = struct.unpack_from('<HH', header, 26)
        offset = entry['offset'] + 30 + name_length + extra_length
        compressed = range_bytes(client, offset, entry['compressed'])
    if entry['compression'] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif entry['compression'] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError('Unsupported ZIP compression.')
    if len(data) != entry['size'] or zlib.crc32(data) & 0xffffffff != entry['crc']:
        raise ValueError('Member size or CRC check failed.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print('Fetched', path, flush=True)
    return {'path': str(path), 'bytes': len(data), 'sha256': sha256(data).hexdigest()}


def fetch_checkpoint(root):
    path = root/'checkpoint_epoch_41_Arch01_v1.pth'
    if path.exists() and file_hash(path) == CHECKPOINT_SHA256:
        return path
    partial = path.with_suffix('.partial')
    params = {'id': CHECKPOINT_ID, 'export': 'download', 'confirm': 't', 'soundleak_part': 'complete'}
    with session() as client, client.get('https://drive.usercontent.google.com/download',
            params=params, stream=True, timeout=60) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise ValueError('Checkpoint server returned HTML instead of model bytes.')
        total = 0
        with partial.open('wb') as destination:
            for chunk in response.iter_content(2 ** 20):
                total += len(chunk)
                if total > CHECKPOINT_SIZE:
                    raise ValueError('Checkpoint exceeds expected size.')
                destination.write(chunk)
    if partial.stat().st_size != CHECKPOINT_SIZE or file_hash(partial) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint hash or size mismatch.')
    partial.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/real/wifi3d')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = fetch_checkpoint(root)
    cache = root/'archive-index.json'
    index = json.loads(cache.read_text()) if cache.exists() else subset_index()
    cache.write_text(json.dumps(index, indent=2))
    names = {f'E01/S05/A05/{kind}/frame{n:03d}.{extension}'
             for n in FRAME_IDS for kind, extension in [('wifi-csi', 'mat'), ('lidar', 'bin')]}
    selected = [entry for entry in index if entry['name'] in names]
    if len(selected) != len(names):
        raise ValueError('Not all requested paired frames are present.')
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda entry: fetch_member(entry, root), selected))
    manifest = {'schema_version': 1, 'dataset': 'MM-Fi', 'dataset_license': 'CC BY-NC 4.0',
                'dataset_home': 'https://ntu-aiot-lab.github.io/mm-fi/',
                'archive': {'id': ARCHIVE_ID, 'bytes': ARCHIVE_SIZE},
                'recording': 'E01/S05/A05', 'frame_ids': list(FRAME_IDS),
                'model_repository': 'https://github.com/Arritmic/csi2pointcloud',
                'model_commit': UPSTREAM_COMMIT, 'checkpoint_file_id': CHECKPOINT_ID,
                'checkpoint_sha256': file_hash(checkpoint), 'files': sorted(files, key=lambda v: v['path'])}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Prepared 6 real paired frames and a verified checkpoint:', root)


if __name__ == '__main__':
    main()
