"""Recording-by-recording acoustic review, independent of detector tuning."""
import json
from pathlib import Path

import streamlit as st

from .public_review import (AUDIBILITY, DEFAULT_DATA, DEFAULT_REPORT, DEFAULT_ROOT, ISSUES, PROPOSAL_LABELS,
                            RECORDING_STATUSES, case_row, export_reviews, load_review_case, prepare_review_case,
                            read_review, read_screen, save_review, source_audio)
from .report import review_spectrogram_figure, ridge_figure
from .hosting import annotation_root, is_hosted


@st.cache_data(max_entries=3)
def review_assets(identity, report_path, report_hash, data_directory, review_root):
    report, current_hash = read_screen(report_path)
    if current_hash != report_hash:
        raise ValueError('The screening report changed. Reload this screen before reviewing.')
    row = case_row(report, identity)
    directory = prepare_review_case(identity, report_path, data_directory, review_root)
    data = load_review_case(directory, row, report_hash)
    return data, source_audio(data_directory, row)


def _move_recording(identities, amount):
    current = st.session_state.get('public_review_recording', identities[0])
    index = identities.index(current) if current in identities else 0
    st.session_state['public_review_recording'] = identities[min(max(index+amount, 0), len(identities)-1)]


def _reload_notes(prefix, baseline_key):
    for key in list(st.session_state):
        if key.startswith(prefix) or key == baseline_key:
            del st.session_state[key]
    st.session_state['public_review_notice'] = 'Saved notes reloaded.'


def render_public_review(report_path=DEFAULT_REPORT, data_directory=DEFAULT_DATA, review_root=DEFAULT_ROOT):
    report_path, review_root = Path(report_path), Path(review_root)
    asset_root = review_root
    if is_hosted() and review_root == DEFAULT_ROOT:
        review_root = annotation_root(DEFAULT_ROOT)
    if not report_path.exists():
        st.info('Prepare the public acoustic screen to review its recordings.')
        st.code('.venv/bin/python -m soundleak.cli public-aircraft-screen --fetch')
        return
    try:
        report, report_hash = read_screen(report_path)
        export, csv_text = export_reviews(report, report_hash, review_root)
    except (ValueError, OSError, KeyError) as exc:
        st.error(str(exc))
        return
    st.caption(f"PUBLIC AUDIO REVIEW · Praveen Pathak · CC BY 4.0 · {len(report['cases'])} published aircraft passages")
    if is_hosted():
        st.caption('Review notes are private to this browser session and temporary. Export your JSON or CSV to keep them.')
    if 'public_review_notice' in st.session_state:
        st.success(st.session_state.pop('public_review_notice'))
    cols = st.columns(4)
    cols[0].metric('Recordings', str(len(report['cases'])))
    cols[1].metric('Reviewed recordings', str(export['reviewed_recordings']))
    cols[2].metric('Reviewed proposals', f"{export['reviewed_proposals']}/{report['total_acoustic_proposals']}")
    cols[3].metric('Recordings without proposals', str(report['cases_without_proposals']))
    st.caption('Listen and inspect each ridge before assigning a label. Human labels are saved separately from the frozen detector results. Save notes before switching recordings.')
    saved = {r['case_id']: r for r in export['reviews']}
    filter_name = st.selectbox('Review queue', ['All recordings', 'No proposals', 'Unreviewed', 'Needs follow-up', 'Reviewed'], key='public_review_filter')
    rows = report['cases']
    if filter_name == 'No proposals':
        rows = [r for r in rows if r.get('proposal_count') == 0 and r['state'] == 'screened']
    elif filter_name != 'All recordings':
        rows = [r for r in rows if saved[r['id']]['recording_status'] == filter_name]
    if not rows:
        st.info('No recordings in this queue. Select another queue to continue reviewing.')
    else:
        identities = [r['id'] for r in rows]
        if st.session_state.get('public_review_recording') not in identities:
            st.session_state['public_review_recording'] = next((r['id'] for r in rows if r.get('proposal_count') == 0), identities[0])
        previous, select, following = st.columns([1, 5, 1], vertical_alignment='bottom')
        index = identities.index(st.session_state['public_review_recording'])
        previous.button('← Previous', disabled=index == 0, on_click=_move_recording, args=(identities, -1), width='stretch')
        identity = select.selectbox('Recording', identities, key='public_review_recording',
                                    format_func=lambda key: f"{key} · {case_row(report, key).get('proposal_count', '—')} proposals · {saved[key]['recording_status']}")
        following.button('Next →', disabled=index == len(identities)-1, on_click=_move_recording, args=(identities, 1), width='stretch')
        row = case_row(report, identity)
        st.subheader(identity)
        st.caption(f"{row['category'].replace('_', ' ')} · recording {identities.index(identity)+1} of {len(identities)} in this queue")
        prefix = f'pr_{report_hash[:12]}_{identity}_'
        baseline_key = prefix+'baseline'
        if baseline_key not in st.session_state:
            st.session_state[baseline_key] = read_review(row, report_hash, review_root)
        baseline, revision = st.session_state[baseline_key]
        data = None
        if row['state'] == 'screened':
            try:
                with st.spinner('Loading verified audio and original-screen ridges…'):
                    data, wav = review_assets(identity, str(report_path), report_hash, str(data_directory), str(asset_root))
                proposals = data['discovery'].proposals
                duration = float(data['discovery'].duration_s)
                if not proposals:
                    st.info('No proposal passed the frozen screen for this recording. Review the audio and spectrogram to assess what was missed.')
                selected = st.selectbox('Inspect proposal', ['All proposals']+[p.id for p in proposals],
                                       format_func=lambda value: value if value == 'All proposals' else f"P{[p.id for p in proposals].index(value)+1} · {value}", key=prefix+'inspect')
                selected_id = None if selected == 'All proposals' else selected
                time_range = st.slider('Display time range · seconds', 0., duration, (0., duration), step=.25, format='%.2f', key=prefix+'time')
                frequency_range = st.slider('Display frequency range · Hz', 40, 800, (40, 800), step=10, key=prefix+'frequency')
                if time_range[1] > time_range[0] and frequency_range[1] > frequency_range[0]:
                    st.audio(wav, format='audio/wav', start_time=time_range[0], end_time=time_range[1])
                    st.pyplot(review_spectrogram_figure(data['time_s'], data['frequency_hz'], data['power_db'], proposals,
                                                       selected_id, time_range, frequency_range), width='stretch')
                else:
                    st.info('Choose a time and frequency range with positive width to display the spectrogram.')
                st.caption('White: retained proposal ridges · Cyan: selected proposal. Display ranges only change the view and playback interval. Ridges were reconstructed using saved settings and checked against the original screening report.')
                if selected_id:
                    proposal = next(p for p in proposals if p.id == selected_id)
                    start, end = proposal.ridge.time_s[[0, -1]]
                    st.write(f"Selected proposal: {start:.2f}–{end:.2f} s · {proposal.members} grouped ridge(s) · tone-fit RMSE {proposal.tone_fit['tone_fit_rmse_hz']:.2f} Hz")
                    st.audio(wav, format='audio/wav', start_time=float(start), end_time=float(end))
                    with st.expander('Selected ridge detail'):
                        st.pyplot(ridge_figure(proposal.ridge, selected_id), width='stretch')
            except (ValueError, OSError, KeyError) as exc:
                st.error(f'Could not load verified review assets: {exc}')
        else:
            st.warning('The original screen failed for this recording: '+row.get('reason', 'Unknown reason'))
        _review_form(row, report_hash, prefix, baseline_key, baseline, revision, review_root)
    with st.expander('Saved review table and exports'):
        st.dataframe([{'Recording': r['case_id'], 'Status': r['recording_status'], 'Aircraft audible': r['aircraft_audible'],
                       'Reviewed proposals': sum(p['label'] != 'Unreviewed' for p in r['proposals'].values()),
                       'Issues': ', '.join(r['issues']), 'Notes': r['notes']} for r in export['reviews']],
                     hide_index=True, width='stretch', height=280)
        left, right = st.columns(2)
        left.download_button('Export review JSON', json.dumps(export, indent=2, allow_nan=False),
                             'public-aircraft-reviews.json', 'application/json', width='stretch')
        right.download_button('Export review CSV', csv_text, 'public-aircraft-reviews.csv', 'text/csv', width='stretch')
        st.caption(f'Local notes: {review_root/report_hash/"annotations"}. Exports retain every unreviewed recording and proposal.')
    st.caption('These are acoustic-review judgments. Aircraft identity and receiver-location accuracy remain unvalidated. Source: https://zenodo.org/records/23424278 · CC BY 4.0.')


def _review_form(row, report_hash, prefix, baseline_key, baseline, revision, review_root):
    st.subheader('Review notes')
    st.button('Reload saved notes', on_click=_reload_notes, args=(prefix, baseline_key), key=prefix+'reload')
    with st.form(prefix+'form'):
        left, right = st.columns(2)
        status = left.selectbox('Recording review status', RECORDING_STATUSES,
                                index=RECORDING_STATUSES.index(baseline['recording_status']), key=prefix+'status')
        audible = right.selectbox('Aircraft audible?', AUDIBILITY,
                                  index=AUDIBILITY.index(baseline['aircraft_audible']), key=prefix+'audible')
        issues = st.multiselect('Potential issues', ISSUES, default=baseline['issues'], key=prefix+'issues')
        notes = st.text_area('Recording notes', value=baseline['notes'], key=prefix+'notes')
        decisions = []
        for i, event in enumerate(row.get('discovery', {}).get('events', []), 1):
            prior = baseline['proposals'][event['id']]
            decisions.append({'Proposal': event['id'], 'Marker': f'P{i}', 'Start · s': round(event['start_s'], 2),
                              'End · s': round(event['end_s'], 2), 'Label': prior['label'],
                              'Related proposal': prior['related_event_id'], 'Notes': prior['notes']})
        if decisions:
            st.caption('Label each retained proposal. For duplicate harmonics, select the related proposal in this recording.')
            decisions = st.data_editor(decisions, hide_index=True, width='stretch', num_rows='fixed',
                disabled=['Proposal', 'Marker', 'Start · s', 'End · s'], key=prefix+'decisions',
                column_config={'Label': st.column_config.SelectboxColumn('Label', options=PROPOSAL_LABELS, required=True),
                               'Related proposal': st.column_config.SelectboxColumn('Related proposal', options=['']+list(baseline['proposals'])),
                               'Notes': st.column_config.TextColumn('Notes')})
        if st.form_submit_button('Save review', type='primary'):
            review = {**baseline, 'recording_status': status, 'aircraft_audible': audible,
                      'issues': issues, 'notes': notes,
                      'proposals': {d['Proposal']: {'label': d['Label'], 'related_event_id': d['Related proposal'] or '',
                                                   'notes': d['Notes'] or ''} for d in decisions}}
            try:
                value, new_revision = save_review(review, row, report_hash, revision, review_root)
                st.session_state[baseline_key] = (value, new_revision)
                st.session_state['public_review_notice'] = f"Review saved for {row['id']}."
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
