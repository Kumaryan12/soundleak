"""Fetch a fixed 96-frame, four-environment MM-Fi evaluation sample with HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from fetch_wifi3d import ROOT, UPSTREAM_COMMIT, CHECKPOINT_SHA256, fetch_checkpoint, subset_index, fetch_member

# Official MM-Fi Google Drive split-folder metadata. No whole archives are fetched.
ARCHIVES = {
    'E01': ('1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B', 22_301_071_119, 'S05'),
    'E02': ('1oIPGmsjDlzQsnTDVzIhYRQq-3BHTxQ8o', 20_886_295_287, 'S15'),
    'E03': ('1WjfPToIpi1a0cRYBvIr2yZoq_2jQpPQq', 18_864_127_686, 'S25'),
    'E04': ('1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1', 20_953_639_005, 'S35'),
}
FRAME_IDS = tuple(range(1, 277, 25))
ACTIONS = ('A05', 'A27')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/real/wifi3d')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fetch_checkpoint(root)
    records, files = [], []
    for scene, (archive_id, archive_size, subject) in ARCHIVES.items():
        recordings = [f'{scene}/{subject}/{action}' for action in ACTIONS]
        cache = root/f'suite-index-{scene.lower()}.json'
        print('Indexing selected recordings in', scene, flush=True)
        index = json.loads(cache.read_text()) if cache.exists() else subset_index(
            tuple(r+'/' for r in recordings), archive_id, archive_size)
        cache.write_text(json.dumps(index, indent=2))
        for recording in recordings:
            names = {f'{recording}/{kind}/frame{n:03d}.{ext}' for n in FRAME_IDS
                     for kind, ext in [('wifi-csi', 'mat'), ('lidar', 'bin')]}
            selected = [entry for entry in index if entry['name'] in names]
            if len(selected) != len(names):
                raise ValueError(f'Missing paired frames for {recording}; no silent sample substitution.')
            with ThreadPoolExecutor(max_workers=3) as pool:
                downloaded = list(pool.map(lambda entry: fetch_member(entry, root, recording,
                                           archive_id, archive_size), selected))
            files.extend(downloaded)
            records.append({'recording': recording, 'frame_ids': list(FRAME_IDS),
                            'environment': scene, 'subject': subject, 'action': recording.split('/')[-1],
                            'archive': {'id': archive_id, 'bytes': archive_size}})
    manifest = {'schema_version': 1, 'dataset': 'MM-Fi', 'dataset_license': 'CC BY-NC 4.0',
                'dataset_home': 'https://ntu-aiot-lab.github.io/mm-fi/',
                'download_index': 'https://drive.google.com/drive/folders/1mzcM2AU7gZl5BAZikEjvdL-G0lm2EGjd',
                'selection': 'Fixed before scoring: E01/S05, E02/S15, E03/S25, E04/S35; actions A05/A27; frames 1..276 by 25.',
                'training_overlap': 'unverified', 'model_repository': 'https://github.com/Arritmic/csi2pointcloud',
                'model_commit': UPSTREAM_COMMIT, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'recordings': records, 'files': sorted(files, key=lambda v: v['path'])}
    path = root/'evaluation-manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    print('Prepared', len(records)*len(FRAME_IDS), 'paired frames:', path, flush=True)


if __name__ == '__main__':
    main()
