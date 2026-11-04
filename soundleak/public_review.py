"""Frozen acoustic-review assets and separately persisted human annotations."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import zipfile

import numpy as np

from .archive import file_sha256
from .audio import Ridge, decode_audio, spectrogram_data
from .discovery import DiscoveryConfig, DiscoveryResult, Proposal, discover

DEFAULT_REPORT = Path('outputs/evaluations/public-aircraft-screen.json')
DEFAULT_DATA = Path('data/real/pathak')
DEFAULT_ROOT = Path('outputs/reviews/public-aircraft')
RECORDING_STATUSES = ('Unreviewed', 'Reviewed', 'Needs follow-up')
AUDIBILITY = ('Not assessed', 'Yes', 'No', 'Unclear')
PROPOSAL_LABELS = ('Unreviewed', 'Follows aircraft tone', 'False proposal', 'Fragmented track', 'Duplicate harmonic', 'Unclear')
ISSUES = ('Weak or obscured tone', 'Fragmented ridge', 'Frequency outside screen bands', 'Little pitch movement',
          'Overlapping sounds', 'Spurious proposals', 'Duplicate harmonics', 'Unclear')
_REVIEW_LOCK = threading.RLock()


def read_screen(path=DEFAULT_REPORT):
    path = Path(path)
    contents = path.read_bytes()
    report = json.loads(contents)
    if report.get('schema_version') != 1 or not isinstance(report.get('cases'), list):
        raise ValueError('Unsupported public screening report.')
    ids = [c['id'] for c in report['cases']]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('Invalid or duplicate recording identifiers in screening report.')
    return report, sha256(contents).hexdigest()


def case_row(report, identity):
    row = next((c for c in report['cases'] if c['id'] == identity), None)
    if row is None:
        raise ValueError('Recording is not part of this screening report.')
    return row


def _scope(root, report_hash):
    if not re.fullmatch('[a-f0-9]{64}', report_hash):
        raise ValueError('Invalid screening report hash.')
    return Path(root)/report_hash


def _matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(_matches(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(_matches(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-8))
    return actual == expected


def source_audio(data_directory, row):
    name = row['archive']
    if Path(name).name != name:
        raise ValueError('Invalid archive filename.')
    with zipfile.ZipFile(Path(data_directory)/name) as archive:
        info = archive.getinfo(row['member'])
        if info.file_size > 100_000_000 or info.file_size != row['uncompressed_bytes']:
            raise ValueError('Unexpected public audio member size.')
        contents = archive.read(row['member'])  # Full read checks the member CRC.
    if sha256(contents).hexdigest() != row['audio_sha256']:
        raise ValueError('Original WAV differs from the frozen acoustic screen.')
    return contents


def prepare_review_case(identity, report_path=DEFAULT_REPORT, data_directory=DEFAULT_DATA, root=DEFAULT_ROOT):
    report, report_hash = read_screen(report_path)
    row = case_row(report, identity)
    if row['state'] != 'screened':
        raise ValueError('This recording failed the original screen; no frozen measurements are available.')
    directory = _scope(root, report_hash)/'assets'/identity
    if (directory/'manifest.json').exists():
        load_review_case(directory, row, report_hash)
        return directory
    for name in ('audio.py', 'discovery.py'):
        if file_sha256(Path(__file__).with_name(name)) != report['analysis_code_sha256'][name]:
            raise ValueError('The detector code has changed. Original review assets must be prepared with the saved code version.')
    contents = source_audio(data_directory, row)
    config = DiscoveryConfig(**report['discovery_config'])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'recording.wav'
        path.write_bytes(contents)
        audio, sr = decode_audio(path, max_duration_s=600)
    result = discover(audio, sr, config)
    actual, expected = result.metadata(), row['discovery']
    for key in ('duration_s', 'scanned_windows', 'raw_proposals', 'config', 'events'):
        if not _matches(actual[key], expected[key]):
            raise ValueError(f'Reconstructed {key} differs from the frozen screen. Refusing to relabel it as the original result.')
    times, frequencies, power = spectrogram_data(audio, sr, 0, result.duration_s, config.bands[0][0], config.bands[-1][1])
    arrays = {'audio': audio, 'sample_rate': sr, 'time_s': times, 'frequency_hz': frequencies, 'power_db': power}
    for proposal in result.proposals:
        for key in ('time_s', 'frequency_hz', 'prominence_db', 'frequency_bins_hz', 'power_db'):
            arrays[proposal.id+'_'+key] = getattr(proposal.ridge, key)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A completed directory is published only after both files have been written.
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary)/identity
        staging.mkdir()
        np.savez_compressed(staging/'measurements.npz', **arrays)
        manifest = {'schema_version': 1, 'case_id': identity, 'screen_report_sha256': report_hash,
                    'source_audio_sha256': row['audio_sha256'], 'measurements_sha256': file_sha256(staging/'measurements.npz'),
                    'discovery': expected, 'analysis_code_sha256': report['analysis_code_sha256'],
                    'reconstruction': 'Original WAV decoded with the saved detector configuration and unchanged audio/discovery code. All saved proposal metadata matched within numerical tolerance. Arrays were reconstructed; the original screening report contained summaries only.'}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
        try:
            staging.rename(directory)
        except OSError:
            if not (directory/'manifest.json').exists():
                raise
            load_review_case(directory, row, report_hash)
    return directory


def load_review_case(directory, row, report_hash):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('case_id') != row['id']
        or manifest.get('screen_report_sha256') != report_hash or manifest.get('source_audio_sha256') != row['audio_sha256']
        or manifest.get('discovery') != row['discovery']):
        raise ValueError('Review assets do not belong to this frozen recording.')
    if file_sha256(directory/'measurements.npz') != manifest['measurements_sha256']:
        raise ValueError('Review measurements failed their integrity check.')
    with np.load(directory/'measurements.npz', allow_pickle=False) as bundle:
        proposals = []
        for meta in row['discovery']['events']:
            ridge = Ridge(*(bundle[meta['id']+'_'+key] for key in
                            ('time_s', 'frequency_hz', 'prominence_db', 'frequency_bins_hz', 'power_db')))
            fit = {k: meta[k] for k in ('tone_fit_rmse_hz', 'trend_explained_fraction', 'fitted_tone_hz', 'approx_cpa_receive_s', 'fit_converged')}
            proposals.append(Proposal(meta['id'], ridge, fit, meta['grouped_proposals']))
        meta = row['discovery']
        discovery = DiscoveryResult(proposals, meta['duration_s'], meta['scanned_windows'], meta['raw_proposals'], meta['config'], meta['elapsed_s'])
        return {'audio': bundle['audio'], 'sample_rate': int(bundle['sample_rate']), 'discovery': discovery,
                'time_s': bundle['time_s'], 'frequency_hz': bundle['frequency_hz'], 'power_db': bundle['power_db']}


def empty_review(row, report_hash):
    return {'schema_version': 1, 'screen_report_sha256': report_hash, 'case_id': row['id'],
            'source_audio_sha256': row.get('audio_sha256'), 'updated_utc': None,
            'recording_status': 'Unreviewed', 'aircraft_audible': 'Not assessed', 'issues': [], 'notes': '',
            'proposals': {p['id']: {'label': 'Unreviewed', 'related_event_id': '', 'notes': ''}
                          for p in row.get('discovery', {}).get('events', [])}}


def validate_review(review, row, report_hash):
    template = empty_review(row, report_hash)
    for key in ('schema_version', 'screen_report_sha256', 'case_id', 'source_audio_sha256'):
        if review.get(key) != template[key]:
            raise ValueError('Review belongs to a different frozen recording.')
    if review.get('recording_status') not in RECORDING_STATUSES or review.get('aircraft_audible') not in AUDIBILITY:
        raise ValueError('Unknown recording review label.')
    if not isinstance(review.get('issues'), list) or any(v not in ISSUES for v in review['issues']):
        raise ValueError('Unknown recording issue.')
    if not isinstance(review.get('notes'), str) or len(review['notes']) > 20000:
        raise ValueError('Review notes must be text of at most 20,000 characters.')
    if not isinstance(review.get('proposals'), dict) or set(review['proposals']) != set(template['proposals']):
        raise ValueError('Review must retain every original proposal identifier.')
    for identity, item in review['proposals'].items():
        if item.get('label') not in PROPOSAL_LABELS:
            raise ValueError('Unknown proposal review label.')
        other = item.get('related_event_id', '')
        if other and (other not in review['proposals'] or other == identity):
            raise ValueError('A related proposal must be another proposal in this recording.')
        if item['label'] == 'Duplicate harmonic' and not other:
            raise ValueError('Select a related proposal when marking a duplicate harmonic.')
        if not isinstance(item.get('notes'), str) or len(item['notes']) > 20000:
            raise ValueError('Proposal notes must be text of at most 20,000 characters.')


def read_review(row, report_hash, root=DEFAULT_ROOT):
    path = _scope(root, report_hash)/'annotations'/(row['id']+'.json')
    if not path.exists():
        return empty_review(row, report_hash), None
    contents = path.read_bytes()
    review = json.loads(contents)
    validate_review(review, row, report_hash)
    return review, sha256(contents).hexdigest()


def save_review(review, row, report_hash, expected_revision, root=DEFAULT_ROOT):
    validate_review(review, row, report_hash)
    path = _scope(root, report_hash)/'annotations'/(row['id']+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with _REVIEW_LOCK:
        revision = file_sha256(path) if path.exists() else None
        if revision != expected_revision:
            raise ValueError('This review changed in another session. Reload the saved review before saving again.')
        value = {**review, 'updated_utc': datetime.now(timezone.utc).isoformat()}
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return value, file_sha256(path)


def export_reviews(report, report_hash, root=DEFAULT_ROOT):
    reviews = [read_review(row, report_hash, root)[0] for row in report['cases']]
    result = {'schema_version': 1, 'screen_report_sha256': report_hash, 'source': report['source'],
              'exported_utc': datetime.now(timezone.utc).isoformat(), 'declared_recordings': len(reviews),
              'reviewed_recordings': sum(r['recording_status'] == 'Reviewed' for r in reviews),
              'reviewed_proposals': sum(p['label'] != 'Unreviewed' for r in reviews for p in r['proposals'].values()),
              'reviews': reviews, 'interpretation': 'Human acoustic-review annotations. Unreviewed items remain explicit. Labels do not establish flight identity or independent localization accuracy.'}
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['case_id', 'recording_status', 'aircraft_audible', 'issues', 'recording_notes',
                                               'proposal_id', 'label', 'related_event_id', 'proposal_notes'])
    writer.writeheader()
    for review in reviews:
        base = {k: review[k] for k in ('case_id', 'recording_status', 'aircraft_audible')}
        base.update(issues='; '.join(review['issues']), recording_notes=review['notes'])
        for identity, item in review['proposals'].items():
            writer.writerow({**base, 'proposal_id': identity, 'label': item['label'], 'related_event_id': item['related_event_id'], 'proposal_notes': item['notes']})
        if not review['proposals']:
            writer.writerow(base)
    return result, stream.getvalue()
