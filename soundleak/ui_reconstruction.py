"""Interactive replay of an offline, real-CSI reconstruction experiment."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import streamlit as st
from streamlit.components.v2 import component

from .reconstruction import read_reconstruction, ply_bytes

REPORT = Path('outputs/wifi3d/mmfi-reference/reconstruction.json')


@st.cache_data(max_entries=2)
def load_report(path, content_hash):
    if sha256(Path(path).read_bytes()).hexdigest() != content_hash:
        raise ValueError('Reconstruction report changed during loading.')
    return read_reconstruction(path)


@lru_cache(maxsize=1)
def cloud_viewer():
    root = Path(__file__).with_name('reconstruction_frontend')
    return component('soundleak_wifi3d_viewer', html=(root/'viewer.html').read_text(),
                     css=(root/'viewer.css').read_text(), js=(root/'viewer.js').read_text())


def render_reconstruction():
    st.info('Real recorded Wi-Fi CSI · Pretrained-model inference · Offline reconstruction replay')
    if not REPORT.exists():
        st.warning('The recorded reconstruction has not been prepared in this checkout.')
        st.code('python scripts/fetch_wifi3d.py\npython scripts/run_wifi3d.py')
        return
    try:
        report = load_report(str(REPORT), sha256(REPORT.read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    st.write('Explore point clouds predicted from six frames of one real MM-Fi Wi-Fi recording. '
             'The model receives CSI amplitude and phase. The amber LiDAR scan is an independently '
             'measured comparison and is introduced after prediction.')
    st.caption('Coordinates are standardized model units. These outputs do not establish physical room dimensions, '
               'a surveyed floor plan or the layout of your home.')
    index = st.select_slider('Recorded frame', options=list(range(len(report['frames']))),
                             format_func=lambda i: f"Frame {report['frames'][i]['frame_number']:03d}", key='wifi3d_frame')
    frame = report['frames'][index]
    cloud_viewer()(data={**frame, 'zero_input_control': report['zero_input_control']},
                   key='wifi3d-point-cloud', height='content')
    st.subheader('How much does the signal help?')
    st.caption('Lower is better: mean bidirectional nearest-point distance, standardized units, with no alignment fitted to the reference.')
    a, b, c = st.columns(3)
    a.metric('Wi-Fi prediction · mean distance', f"{report['mean_nn_distance']:.4f}")
    b.metric('Zero-input control', f"{report['mean_zero_input_nn_distance']:.4f}")
    c.metric('Fixed first-frame prediction', f"{report['mean_fixed_frame_nn_distance']:.4f}")
    if report['mean_fixed_frame_nn_distance'] <= report['mean_nn_distance']:
        st.warning('In this six-frame sample, reusing the first prediction performs slightly better than '
                   'updating the prediction from each Wi-Fi frame. The model runs on real signals, but this '
                   'experiment does not demonstrate reliable tracking of changing scene geometry.')
    st.write('A trained model can produce a plausible point cloud even with zero input. Use the zero-input '
             'control view and the measured reference to assess the result rather than judging the 3D animation alone.')
    left, right = st.columns(2)
    left.download_button('Download predicted point cloud · PLY', ply_bytes(frame['predicted']),
                         f"mmfi-frame{frame['frame_number']:03d}-prediction.ply", 'application/octet-stream', on_click='ignore')
    right.download_button('Download reconstruction report', json.dumps(report, allow_nan=False),
                          'wifi3d-reconstruction.json', 'application/json', on_click='ignore')
    with st.expander('Sources, measurements and limitations'):
        st.markdown('[CSI2PointCloud model](https://github.com/Arritmic/csi2pointcloud) · '
                    '[MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi/) · '
                    '[Reproduction guide](https://github.com/Kumaryan12/soundleak/blob/main/docs/wifi3d.md)')
        st.write('Dataset: MM-Fi, Jianfei Yang and colleagues (NeurIPS 2023), CC BY-NC 4.0. '
                 'Model: Määttä, Sharifipour, Bordallo López and Álvarez Casado, CSI2PointCloud (2025), MIT code.')
        for limitation in report['limitations']:
            st.caption(limitation)
        st.code('Model source commit: ' + report['model_commit'] + '\nCheckpoint SHA256: ' + report['checkpoint_sha256'])
