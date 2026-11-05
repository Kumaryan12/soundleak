"""Recorded signal-dependence evaluation, playback and surface-error inspection."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .reconstruction import ply_bytes
from .reconstruction_eval import read_evaluation, point_errors, METHODS

INDEX = Path('outputs/wifi3d/evaluation/evaluation.json')


@st.cache_data(max_entries=2)
def load_evaluation(path, content_hash):
    if sha256(Path(path).read_bytes()).hexdigest() != content_hash:
        raise ValueError('Evaluation index changed during loading.')
    return read_evaluation(path)


def render_evaluation(viewer):
    if not INDEX.exists():
        st.info('Prepare the evaluation sample to enable this lab.')
        st.code('python scripts/fetch_wifi3d_suite.py\npython scripts/evaluate_wifi3d.py')
        return
    try:
        report, clouds = load_evaluation(str(INDEX), sha256(INDEX.read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Evaluation could not be loaded: {exc}')
        return
    summary = report['summary']
    methods = summary['available_methods']
    st.info(f"Recorded Wi-Fi evaluation · {summary['environments']} environment(s) · Predetermined negative controls · Training overlap unverified")
    st.write('Test whether matching Wi-Fi signals improve the reconstruction beyond the model’s learned scene prior. '
             'Every prediction was computed before its measured LiDAR reference was loaded.')
    a, b, c = st.columns(3)
    a.metric('Recorded environments', summary['environments'])
    b.metric('Paired CSI / LiDAR frames', summary['frames'])
    c.metric('Negative controls available', f"{len(methods)-1} / {len(METHODS)-1}")
    if summary['missing_methods']:
        missing = ', '.join(METHODS[k] for k in summary['missing_methods'])
        st.warning('Expanded sample pending: '+report['expansion_status']['reason']+' Unavailable control: '+missing+'. '
                   'Only the verified downloaded sample is scored below; cross-environment performance has not been measured.')
    if summary['correct_beats_all_controls']:
        st.info('Correct CSI has lower average shape error than each control on this selected sample. '
                'Checkpoint overlap is unverified, so this is not evidence of held-out reconstruction accuracy.')
    else:
        matching = ', '.join(METHODS[k] for k in methods if k != 'correct' and
                             summary['mean_distance'][k] <= summary['mean_distance']['correct'])
        st.warning((f'Correct CSI does not beat every available control on average. Matching or better controls: {matching}. '
                    if matching else 'Correct CSI beats the available controls, but the required different-environment control is missing. ')+
                   'This run does not establish reliable signal-dependent reconstruction.')
    tabs = st.tabs(['Compare and play', 'Scores and changes', 'Training overlap and sources'])
    with tabs[0]:
        names = list(clouds)
        name = st.selectbox('Recorded sequence', names, key='wifi3d_eval_sequence')
        record = next(r for r in report['records'] if r['recording'] == name)
        method = st.selectbox('Input to compare', methods, format_func=METHODS.get, key='wifi3d_eval_method')
        st.caption(report['control_design'][method])
        all_frames = {f['id']: f for cloud in clouds.values() for f in cloud['frames']}
        view_points = [clouds[name]['zero_input_control']]
        for row in record['frames']:
            view_points.extend([all_frames[row['id']]['predicted'], all_frames[row['id']]['reference']])
            if row['unrelated_source'] is not None:
                view_points.append(all_frames[row['unrelated_source']]['predicted'])
        view_points = np.concatenate(view_points)
        view_bounds = {'low': view_points.min(axis=0).tolist(), 'high': view_points.max(axis=0).tolist()}
        color_limit = st.slider('Error color limit · standardized units', .05, 1., .25, .05,
                                help='Surface distances at or above this value share the hottest color. This is not confidence or meters.')
        playing = st.toggle('Play recorded frames', key='wifi3d_eval_play')
        st.caption('Playback advances sampled frame indices every 1.25 seconds and loops. It does not reproduce the recording’s real-time speed.')

        @st.fragment(run_every=1.25 if playing else None)
        def frame_view():
            key = 'wifi3d_eval_frame_'+name.replace('/', '_')
            if key not in st.session_state:
                st.session_state[key] = 0
            elif playing:
                st.session_state[key] = (st.session_state[key]+1) % len(record['frames'])
            index = st.select_slider('Evaluation frame', range(len(record['frames'])),
                       format_func=lambda i: f"Frame {record['frames'][i]['frame_number']:03d}", key=key)
            row = record['frames'][index]
            target = all_frames[row['id']]
            source = row['id']
            if method == 'fixed':
                source = record['frames'][0]['id']
            elif method == 'shuffled':
                source = row['shuffled_source']
            elif method == 'unrelated':
                source = row['unrelated_source']
            predicted = clouds[name]['zero_input_control'] if method == 'zero' else all_frames[source]['predicted']
            p_errors, r_errors = point_errors(predicted, target['reference'])
            st.caption('Target: '+row['id']+' · Input: '+('all-zero tensor' if method == 'zero' else source))
            viewer()(data={**target, 'predicted': predicted, 'zero_input_control': clouds[name]['zero_input_control'],
                           'method_label': METHODS[method], 'prediction_errors': p_errors.tolist(),
                           'reference_errors': r_errors.tolist(), 'error_limit': color_limit, 'view_bounds': view_bounds},
                     key='wifi3d-evaluation-cloud', height='content')
            x, y, z = st.columns(3)
            x.metric('Symmetric surface distance', f"{row['scores'][method]:.4f}")
            y.metric('Predicted points · 90th percentile error', f'{np.percentile(p_errors, 90):.4f}')
            z.metric('Measured points · 90th percentile gap', f'{np.percentile(r_errors, 90):.4f}')
            st.caption('Surface error colors show local nearest-point distance on both clouds. '
                       'Reference-side gaps can reveal measured surfaces missing from the prediction. '
                       'Coordinates are standardized per the replay convention; no fitted alignment or physical room dimensions.')
            st.download_button('Download displayed prediction · PLY', ply_bytes(predicted),
                               name.replace('/', '-')+f"-frame{target['frame_number']:03d}-{method}.ply",
                               'application/octet-stream', on_click='ignore', key='wifi3d_eval_ply')
        frame_view()
    with tabs[1]:
        st.subheader('Shape error by input')
        st.caption('Mean bidirectional nearest-point distance, standardized units; lower is better. '
                   'Each recording receives equal aggregate weight. Frames are correlated, not independent trials.')
        scores = pd.DataFrame([{'Input': METHODS[k], 'Mean shape error': summary['mean_distance'][k],
                                'Mean change-field error': summary['mean_change_field_mae'][k]} for k in methods])
        st.dataframe(scores, hide_index=True, width='stretch')
        st.bar_chart(scores.set_index('Input')[['Mean shape error']], color='#51d6ca')
        table = [{'Recording': r['recording'], **{METHODS[k]: r['mean_distance'][k] for k in methods}}
                 for r in report['records']]
        st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
        st.subheader('Error across sampled frames')
        selected = st.selectbox('Sequence for error and change plots', names, key='wifi3d_eval_plot_sequence')
        chosen = next(r for r in report['records'] if r['recording'] == selected)
        frame_scores = pd.DataFrame([{'Frame': r['frame_number'], **{METHODS[k]: r['scores'][k] for k in methods}}
                                     for r in chosen['frames']]).set_index('Frame')
        st.line_chart(frame_scores, x_label='Recorded frame index', y_label='Shape error · standardized units')
        st.subheader('Does predicted geometry change with the reference?')
        st.caption('At the same 1,728 grid probes, compare the signed change in distance to the nearest surface. '
                   'Lower change-field MAE is better. A fixed prediction has zero predicted change. '
                   'These scores describe standardized surface changes, not motion direction or displacement in meters.')
        st.write(f"Correct CSI beats the fixed prediction’s temporal error in "
                 f"{summary['records_beating_fixed_temporal']} of {summary['recordings']} recordings.")
        transitions = chosen['transitions']
        magnitude = pd.DataFrame([{'To frame': t['to_frame'], 'Measured reference': t['reference_change'],
                                  'Correct CSI prediction': t['methods']['correct']['predicted_change'],
                                  'Fixed prediction': t['methods']['fixed']['predicted_change']} for t in transitions]).set_index('To frame')
        st.line_chart(magnitude, x_label='Transition ending at frame', y_label='Mean absolute surface-field change')
        change_errors = pd.DataFrame([{'To frame': t['to_frame'], **{METHODS[k]: t['methods'][k]['change_field_mae']
                                      for k in methods}} for t in transitions]).set_index('To frame')
        st.line_chart(change_errors, x_label='Transition ending at frame', y_label='Change-field MAE · lower is better')
        temporal_table = [{'Recording': r['recording'], **{METHODS[k]: r['mean_change_field_mae'][k] for k in methods}}
                          for r in report['records']]
        st.dataframe(pd.DataFrame(temporal_table), hide_index=True, width='stretch')
        st.download_button('Download all frame scores · CSV', (INDEX.parent/'frame-scores.csv').read_bytes(),
                           'wifi3d-frame-scores.csv', 'text/csv', on_click='ignore')
        st.download_button('Download evaluation report · JSON', json.dumps(report, allow_nan=False),
                           'wifi3d-evaluation.json', 'application/json', on_click='ignore')
        if (INDEX.parent/'evaluation.png').exists():
            st.download_button('Download evaluation figure · PNG', (INDEX.parent/'evaluation.png').read_bytes(),
                               'wifi3d-evaluation.png', 'image/png', on_click='ignore')
    with tabs[2]:
        audit = report['training_overlap']
        st.warning('Checkpoint training overlap: UNVERIFIED. '+audit['reason'])
        st.write(audit['note'])
        st.caption('The table describes hypothetical membership under configurations at the pinned source commit. '
                   'It does not establish the training history of the downloaded weights.')
        st.dataframe(pd.DataFrame(audit['records']), hide_index=True, width='stretch')
        st.markdown(f"[Pinned configuration]({audit['config_source']}) · [Dataset loader]({audit['loader_source']}) · "
                    f"[Checkpoint writer]({audit['checkpoint_writer_source']})")
        st.code('Checkpoint fields: '+', '.join(audit['checkpoint_fields'])+'\nConfig SHA256: '+audit['config_sha256'])
        st.write('Sample selection: '+report['input_manifest']['selection'])
        for limitation in report['limitations']:
            st.caption('• '+limitation)
        st.markdown('[CSI2PointCloud model](https://github.com/Arritmic/csi2pointcloud) · '
                    '[MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi/) · '
                    '[Reproduction guide](https://github.com/Kumaryan12/soundleak/blob/main/docs/wifi3d-evaluation.md)')
        st.write('MM-Fi: Jianfei Yang and colleagues (NeurIPS 2023), CC BY-NC 4.0, noncommercial research. '
                 'CSI2PointCloud: Määttä, Sharifipour, Bordallo López and Álvarez Casado (2025), MIT architecture code.')
