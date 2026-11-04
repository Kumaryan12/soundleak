"""Synchronized recorded replay and user-triggered local sound analysis."""
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import streamlit as st
from streamlit import runtime
from streamlit.components.v2 import component

from .archive import file_sha256
from .audio import decode_audio
from .discovery import discover
from .live import cumulative_replay, replay_payload
from .hosting import is_hosted

REPLAY_DIRECTORY = Path('outputs/live/regional')


@lru_cache(maxsize=1)
def dashboard():
    folder = Path(__file__).with_name('live_frontend')
    return component('soundleak_live_dashboard', html=(folder/'dashboard.html').read_text(),
                     css=(folder/'dashboard.css').read_text(), js=(folder/'dashboard.js').read_text())


@st.cache_data(max_entries=2)
def synthetic_replay():
    from .demo import make_demo
    demo = make_demo()
    discovery = discover(demo.audio, demo.sample_rate)
    replay = cumulative_replay(discovery.proposals, demo.tracks, np.arange(-3000, 3001, 200), np.arange(-3000, 3001, 200),
                               [-2., 0., 2.], demo.truth_m[2], origin=demo.origin)
    payload = replay_payload(demo.audio, demo.sample_rate, discovery, demo.tracks, demo.origin, replay)
    payload.update({'name': 'Two flyovers · synthetic experiment', 'source_kind': 'synthetic',
                    'scope': 'Synthetic audio and flight geometry. Precomputed proposals appear as complete observations. Reference position is held outside the fit.'})
    # The estimator never receives the reference. Attach it only after fitting.
    payload['reference'] = {'east_m': demo.truth_m[0], 'north_m': demo.truth_m[1], 'label': 'Synthetic recorder reference',
                            'note': 'A synthetic consistency check, not real-world accuracy.'}
    return payload, demo.audio, demo.sample_rate


@st.cache_data(max_entries=2)
def saved_replay(path, content_hash):
    payload = json.loads(Path(path).read_text())
    if file_sha256(path) != content_hash or payload.get('schema_version') != 1:
        raise ValueError('Recorded replay changed or has an unsupported schema.')
    root = Path(payload['source_case'])
    if file_sha256(root/'report.json') != payload['source_report_sha256'] or file_sha256(root/'audio.npz') != payload['source_audio_npz_sha256']:
        raise ValueError('The replay source differs from its frozen case.')
    with np.load(root/'audio.npz', allow_pickle=False) as bundle:
        audio, sr = bundle['audio'], int(bundle['sample_rate'])
    from .real import PUBLIC_REFERENCE
    from .physics import local_xy
    x, y = local_xy(*PUBLIC_REFERENCE, *payload['origin'])
    payload['reference'] = {'east_m': float(x), 'north_m': float(y), 'label': 'Approximate public camera marker',
                            'note': 'Rounded source-informed reference; physical microphone uncertainty is unknown.'}
    return payload, audio, sr


@st.cache_data(max_entries=2)
def playback_wav(audio, sample_rate):
    output = io.BytesIO()
    wavfile.write(output, sample_rate, (np.clip(audio, -1, 1)*32767).astype(np.int16))
    return output.getvalue()


@st.cache_resource
def sound_model(manifest_hash):
    from .sound_events import load_sound_model
    return load_sound_model()


@st.cache_data(max_entries=2)
def example_sound(name):
    transport = {'Car horn · ESC-50': 'car_horn', 'Train · ESC-50': 'train', 'Alarm clock · ESC-50': 'clock_alarm'}
    if name in transport:
        folder = Path('data/real/sound-examples')
        info = json.loads((folder/'manifest.json').read_text())
        clip = next(c for c in info['clips'] if c['category'] == transport[name])
        path = folder/clip['local_file']
        if file_sha256(path) != clip['sha256']:
            raise ValueError('Sound example failed its input-integrity check.')
        return path.read_bytes(), path.name
    if name == 'Cat · public tutorial sample':
        path = Path('data/real/models/yamnet-onnx/samples/miaow_16k.wav')
        return path.read_bytes(), path.name
    from .public_review import read_screen, case_row, source_audio, DEFAULT_DATA
    report, _ = read_screen()
    return source_audio(DEFAULT_DATA, case_row(report, 'N373DF_R06')), 'public-aircraft.wav'


def render_live():
    replay_tab, sound_tab, info_tab = st.tabs(['Recorded replay', 'Record / upload a sound', 'About this experiment'])
    with replay_tab:
        sources = ['Synthetic · two flyovers']
        path = REPLAY_DIRECTORY/'replay.json'
        if path.exists():
            sources.insert(0, 'Real recording · regional catalog')
        source = st.selectbox('Replay source', sources, key='live_replay_source')
        try:
            with st.spinner('Loading synchronized replay…'):
                if source.startswith('Real'):
                    payload, audio, sr = saved_replay(str(path), file_sha256(path))
                else:
                    payload, audio, sr = synthetic_replay()
            wav = playback_wav(audio, sr)
            url = runtime.get_instance().media_file_mgr.add(wav, 'audio/wav', 'soundleak-live-'+payload['source_kind'])
            data = {**payload, 'audio_url': url}
            dashboard()(data=data, key='live-dashboard-'+payload['source_kind'], height='content')
            with st.expander('Replay assumptions and result download'):
                st.write(payload['scope'])
                st.write('Each map uses only the completed groups listed at that playback time. A single shared clock is fitted across those groups. '
                         'The detector originally analyzed the recording offline; playback is a visualization of those observations, not causal live detection. '
                         'The reference is added after fitting and appears only when you reveal it.')
                st.caption(f"{len(payload['states'])-1} observation updates · {payload['evaluated_models']:,} model comparisons · "
                           f"{payload['settings']['grid_spacing_m']:.0f} m grid · {len(payload['clock_offsets_s'])} shared-clock candidates · "
                           'playback uses the mono analysis audio')
                st.download_button('Download replay results', json.dumps(payload, allow_nan=False), 'soundleak-replay.json',
                                   'application/json', on_click='ignore')
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
    with sound_tab:
        render_sound_capture()
    with info_tab:
        st.write('Soundleak Live combines a playback clock, a scrolling spectrogram, reported aircraft positions, and cumulative Doppler fits. '
                 'The real replay uses an existing source-informed recording. The synthetic option keeps a generated reference outside the estimator.')
        st.write('The sound inspector is a separate pretrained sound-event model. It can suggest familiar audible categories such as vehicles, '
                 'animals, speech, alarms and machinery. An object must make a detectable sound, and the model must recognize that sound. '
                 'Sound labels do not establish a particular object, its distance or its location.')
        st.write('Continuous microphone localization and fresh live flight lookup remain future work. The capture control records a sample and analysis runs after you stop. '
                 'Browser microphone access works on localhost or a secure HTTPS origin, subject to your browser’s permission prompt.')
        st.markdown('Sound model: [Google YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet), '
                    '[pinned community ONNX conversion](https://huggingface.co/audiomagic/yamnet-onnx). Inference runs on the app server.')


def render_sound_capture():
    from .sound_events import MODEL_DIRECTORY, classify_sound
    st.subheader('What can you hear?')
    st.caption('Record at least four seconds, then analyze up to two minutes. Location is not inferred from these general sound labels.')
    st.caption('Recordings are uploaded to this app server for analysis, without an external inference API.' if is_hosted()
               else 'Sound suggestions run on this computer, without an external inference API.')
    sources = ['Microphone', 'Upload audio/video']
    if Path('data/real/models/yamnet-onnx/samples/miaow_16k.wav').exists():
        sources.append('Example sounds')
    source = st.radio('Sound input', sources, horizontal=True, key='live_sound_source')
    media, contents, name = None, None, 'recording.wav'
    if source == 'Microphone':
        media = st.audio_input('Record a sound sample', sample_rate=16000, key='live_microphone')
    elif source == 'Upload audio/video':
        media = st.file_uploader('Audio or video sample', type=['wav','webm','mp3','m4a','mp4','ogg','flac','mov'], key='live_sound_upload')
    else:
        names = ['Cat · public tutorial sample']
        if Path('outputs/evaluations/public-aircraft-screen.json').exists():
            names.append('Aircraft · public benchmark sample')
        if Path('data/real/sound-examples/manifest.json').exists():
            names.extend(['Car horn · ESC-50', 'Train · ESC-50', 'Alarm clock · ESC-50'])
        sample = st.selectbox('Example recording', names)
        try:
            contents, name = example_sound(sample)
            if sample.endswith('ESC-50'):
                st.caption('Source: ESC-50, Karol J. Piczak / Freesound contributors · dataset CC BY-NC 3.0 · source IDs retained in the local example manifest')
            else:
                st.caption('Source: Google AudioSet tutorial sample' if sample.startswith('Cat') else 'Source: Praveen Pathak, CC BY 4.0')
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
    manifest = MODEL_DIRECTORY/'manifest.json'
    if not manifest.exists():
        st.info('Install the optional sound model to enable multi-category suggestions.')
        st.code(".venv/bin/python -m pip install -e '.[classify]'\n.venv/bin/python -m soundleak.cli fetch-sound-model")
        return
    if media is not None:
        contents, name = media.getvalue(), media.name
    identity = sha256(contents).hexdigest() if contents else None
    if contents and len(contents) > 50_000_000:
        st.error('Choose a sample under 50 MB.')
        return
    if st.button('Analyze sound', type='primary', disabled=not contents):
        try:
            with st.spinner('Classifying sounds and checking for moving tones…'):
                with tempfile.TemporaryDirectory() as folder:
                    suffix = Path(name).suffix or '.wav'
                    recording = Path(folder)/('sample'+suffix)
                    recording.write_bytes(contents)
                    audio, sr = decode_audio(recording, sample_rate=16000, max_duration_s=120)
                result = classify_sound(audio, sr, sound_model(file_sha256(manifest)))
                low_audio = resample_poly(audio, 32, 125).astype(np.float32)
                discovery = discover(low_audio, 4096)
                result.update({'input_sha256': identity, 'decode_limit_s': 120,
                               'moving_tone_proposals': discovery.metadata(),
                               'location_status': 'not_estimated'})
                st.session_state['live_sound_result'] = {'input_sha256': identity, 'report': result, 'audio': audio, 'sample_rate': sr}
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(str(exc))
    saved = st.session_state.get('live_sound_result')
    if saved and identity == saved['input_sha256']:
        result = saved['report']
        st.audio(playback_wav(saved['audio'], saved['sample_rate']), format='audio/wav')
        st.write(f"Analyzed {result['duration_s']:.1f} seconds · {len(result['moving_tone_proposals']['events'])} moving-tone proposal(s)")
        st.subheader('Suggested sound categories')
        st.dataframe([{'Sound': row['label'], 'Peak model score': round(row['score'], 4),
                       'Mean model score': round(row['mean_score'], 4), 'Peak window · s': round(row['start_s'], 2)} for row in result['peaks']],
                     hide_index=True, width='stretch')
        st.caption('Model scores are not calibrated probabilities. Several labels can apply to the same sound. Moving tones are unconfirmed and are not automatically identified as aircraft.')
        with st.expander('Sound timeline'):
            st.write(f"Label intervals above a {result['display_threshold']:.2f} model-score threshold. Overlapping labels can describe the same source.")
            st.dataframe([{'Sound': row['label'], 'Start · s': round(row['start_s'], 2), 'End · s': round(row['end_s'], 2),
                           'Peak score': round(row['peak_score'], 4)} for row in result['label_intervals']],
                         hide_index=True, width='stretch', height=220)
            st.caption('Frame-by-frame suggestions')
            st.dataframe([{'Start · s': round(row['start_s'], 2), 'End · s': round(row['end_s'], 2), 'Suggestion': row['label'],
                           'Top score': round(row['top'][0]['score'], 4)} for row in result['frames']],
                         hide_index=True, width='stretch', height=280)
        st.download_button('Download sound analysis', json.dumps(result, indent=2, allow_nan=False),
                           'sound-analysis.json', 'application/json', on_click='ignore')
