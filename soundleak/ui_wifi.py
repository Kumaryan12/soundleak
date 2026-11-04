"""Wi-Fi link diagnostics using the user's own RSSI captures."""
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

import streamlit as st

from .hosting import is_hosted
from .wifi import analyze_wifi, read_wifi_csv, synthetic_wifi_csv

ROOT = Path(__file__).resolve().parents[1]


def collector_bundle():
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ('capture_wifi.py', 'capture_wifi.swift'):
            bundle.write(ROOT/'scripts'/name, name)
        bundle.write(ROOT/'docs/wifi.md', 'README.md')
    return output.getvalue()


def render_wifi():
    st.info('Experimental RSSI measurements · Human presence and location are not estimated.')
    st.write('Measure how the Wi-Fi link between your stationary Mac and router varies. '
             'This works with the Mac’s signal-strength readings; it does not collect CSI.')
    st.caption('The capture tool reads the connected link only. It does not scan networks or record network names, '
               'device addresses, passwords or packet contents.')
    with st.expander('Capture your Mac’s Wi-Fi signal', expanded=True):
        st.write('Download and unzip the collector. In Terminal, open that folder and run:')
        st.code('python3 capture_wifi.py --duration 60 --output wifi-capture.csv', language='bash')
        st.write('Keep the Mac and router stationary. Use the first 10 seconds as a static reference, '
                 'then try walking through the area. Upload the CSV below to inspect the measurements. '
                 'The collector needs macOS and Apple’s Swift command-line tools.')
        st.download_button('Download Mac Wi-Fi collector', collector_bundle(), 'soundleak-wifi-collector.zip',
                           'application/zip', on_click='ignore')
        if is_hosted():
            st.caption('This website runs on a cloud server. The local collector supplies your Mac’s measurements; '
                       'the website cannot read them directly. Uploaded CSVs are analyzed on this server.')
        st.markdown('[How this experiment works](https://github.com/Kumaryan12/soundleak/blob/main/docs/wifi.md)')
    choices = ['Upload capture', 'Synthetic demonstration']
    if not is_hosted() and platform.system() == 'Darwin' and shutil.which('swift'):
        choices.append('Capture this Mac · 30 seconds')
    source = st.radio('Wi-Fi input', choices, horizontal=True, key='wifi_input')
    raw = None
    synthetic = source == 'Synthetic demonstration'
    if synthetic:
        raw = synthetic_wifi_csv()
        st.warning('SYNTHETIC DATA · Generated signal variation illustrates the analysis. No person was sensed.')
        st.download_button('Download synthetic CSV', raw, 'synthetic-wifi.csv', 'text/csv', on_click='ignore')
    elif source.startswith('Capture this Mac'):
        if st.button('Start 30-second Wi-Fi capture', type='primary'):
            try:
                with st.spinner('Reading this Mac’s Wi-Fi link for 30 seconds. Keep the first 10 seconds static…'):
                    result = subprocess.run(['swift', str(ROOT/'scripts/capture_wifi.swift'), '--duration', '30'],
                                            capture_output=True, timeout=120)
                if result.returncode:
                    st.error(result.stderr.decode(errors='replace')[:2000])
                else:
                    st.session_state['wifi_local_capture'] = result.stdout
            except (OSError, subprocess.TimeoutExpired) as exc:
                st.error(f'Wi-Fi collection failed: {exc}')
        raw = st.session_state.get('wifi_local_capture')
        if raw:
            st.download_button('Download local capture CSV', raw, 'wifi-capture.csv', 'text/csv', on_click='ignore')
    else:
        upload = st.file_uploader('Wi-Fi capture CSV · maximum 2 MB', type=['csv'], key='wifi_upload')
        if upload is not None:
            if upload.size > 2_000_000:
                st.error('Wi-Fi captures are limited to 2 MB.')
                return
            raw = upload.getvalue()
    if raw is None:
        st.caption('No measurements loaded. Upload a capture or explicitly choose the synthetic demonstration.')
        return
    try:
        capture = read_wifi_csv(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not synthetic and capture.declared_source == 'synthetic-rssi':
        st.warning('This uploaded file declares a synthetic source. It is not a real-world sensing result.')
    st.subheader('Measured link signal')
    st.line_chart({'Time (s)': capture.time_s, 'RSSI (dBm)': capture.rssi_dbm},
                  x='Time (s)', y='RSSI (dBm)')
    reference_s = st.number_input('Static reference duration · seconds', min_value=5., max_value=60.,
                                 value=10., step=1., key='wifi_reference_s')
    confirmed = synthetic or st.checkbox('The Mac and router stayed stationary and the first reference period '
                                         'represents the static link I want to compare against.', key='wifi_reference_confirmed')
    if not confirmed:
        st.caption('Confirm the reference conditions to compare variability. A static reference does not prove a room was empty.')
        return
    try:
        report = analyze_wifi(capture, baseline_s=reference_s)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.subheader('Signal variation relative to reference')
    a, b, c = st.columns(3)
    a.metric('Valid measurements', f"{report['valid_samples']} / {report['samples']}")
    b.metric('Elevated-variation intervals', len(report['events']))
    c.metric('Variability threshold', f"{report['threshold_db']:.2f} dB")
    st.caption('The score is a trailing 3-second RSSI standard deviation. The threshold is derived from the '
               'reference with a 2 dB minimum margin; it is a diagnostic heuristic, not a trained detector.')
    st.line_chart({'Time (s)': [f['time_s'] for f in report['frames']],
                   'Variability (dB)': [f['variability_db'] for f in report['frames']],
                   'Threshold (dB)': [report['threshold_db']] * report['samples']}, x='Time (s)')
    status = report['latest_status'].replace('_', ' ')
    st.write(f'Latest measurement status: **{status}**.')
    if report['events']:
        st.dataframe(report['events'], hide_index=True, width='stretch')
    else:
        st.write('No sustained threshold exceedance was found. This does not establish that no one was present.')
    for warning in report['warnings']:
        st.caption(warning)
    with st.expander('Inspect sample timeline'):
        st.dataframe(report['frames'], hide_index=True, width='stretch')
    st.download_button('Download Wi-Fi analysis', json.dumps(report, indent=2, allow_nan=False),
                       'wifi-analysis.json', 'application/json', on_click='ignore')
