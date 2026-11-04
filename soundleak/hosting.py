"""Runtime paths for a hosted, session-isolated research demo."""
import os
from pathlib import Path
import tempfile
from uuid import uuid4


def is_hosted():
    return os.environ.get('SOUNDLEAK_HOSTED') == '1'


def session_workspace(state=None):
    """Keep hosted user-created files outside the shared bundled examples."""
    if state is None:
        import streamlit as st
        state = st.session_state
    key = '_soundleak_workspace_id'
    if key not in state:
        state[key] = uuid4().hex
    identity = state[key]
    if not isinstance(identity, str) or len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('Invalid session workspace identity.')
    root = Path(os.environ.get('SOUNDLEAK_SESSION_DIRECTORY', tempfile.gettempdir()))/'soundleak-sessions'/identity
    root.mkdir(parents=True, exist_ok=True)
    return root


def case_output_root():
    return session_workspace()/'cases' if is_hosted() else Path('outputs/cases')


def saved_case_reports():
    roots = [Path('outputs/cases')]
    if is_hosted():
        roots.append(case_output_root())
    return sorted({p for root in roots for p in root.glob('*/report.json') if (p.parent/'manifest.json').exists()}, reverse=True)


def annotation_root(default):
    return session_workspace()/'public-review' if is_hosted() else Path(default)
