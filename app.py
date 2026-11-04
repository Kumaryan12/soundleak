"""Run with: .venv/bin/streamlit run app.py"""
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st

from soundleak.audio import decode_audio, extract_ridge
from soundleak.demo import make_demo, tracks_csv, wav_bytes
from soundleak.physics import read_tracks
from soundleak.report import event_figure, heatmap_figure, summary, json_text, png_bytes
from soundleak.search import Event, search
from soundleak.real import run_reference, PUBLIC_REFERENCE, ORIGIN
from soundleak.physics import local_xy
from soundleak.hosting import is_hosted

st.set_page_config(page_title="Soundleak · Doppler lab", page_icon="◉", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.2rem; max-width: 1500px;}
h1 {letter-spacing: -.045em; font-weight: 650 !important;}
[data-testid="stMetric"] {background: #111b29; border: 1px solid #26364a; border-radius: 12px; padding: 16px;}
[data-testid="stMetricLabel"] {color: #93a6bc;}
.eyebrow {font-size: 12px; color: #51e0d2; letter-spacing: .18em; font-weight: 600;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def demo_data():
    return make_demo()


@st.cache_data
def demo_run(use_two, spacing):
    demo = demo_data()
    windows = demo.windows if use_two else demo.windows[:1]
    events = [Event(demo.tracks[n], extract_ridge(demo.audio, demo.sample_rate, a, b, lo, hi)) for n, a, b, lo, hi in windows]
    grid = np.arange(-3000, 3001, spacing)
    result = search(events, grid, grid, demo.truth_m[2])
    return events, result


@st.cache_data
def real_run():
    return run_reference()


def show_results(events, result, origin, altitude, truth=None):
    report = summary(result, events, origin, altitude, truth)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frequency fit · RMSE", f"{result.best_rmse_hz:.3f} Hz")
    m2.metric("Grid spacing", f"{report['grid_spacing_m']:.0f} m")
    m3.metric("Flyovers matched", str(len(events)))
    if truth is not None:
        m4.metric("Error vs synthetic truth", f"{report['synthetic_position_error_m']:.0f} m")
    else:
        m4.metric("Clock adjustment", f"{result.clock_offset_s:+.1f} s")
    st.write("")
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("The acoustic fingerprint")
        st.caption("White: measured tonal ridge · Cyan: prediction at the best grid location")
        for event, prediction in zip(events, result.predictions_hz):
            st.pyplot(event_figure(event, prediction), width="stretch")
    with right:
        st.subheader("Candidate listening positions")
        st.caption("Lower residuals indicate a closer frequency match. Dashed lines show flight paths.")
        heatmap = heatmap_figure(result, events, truth)
        st.pyplot(heatmap, width="stretch")
        st.caption(f"Best grid fit: {report['latitude']:.5f}°, {report['longitude']:.5f}° · "
                   f"{result.best_x_m:+.0f} m east, {result.best_y_m:+.0f} m north of center")
        if truth is None:
            st.info("A good curve fit is a hypothesis. These residuals are not calibrated location probabilities.")
        st.download_button("Download result JSON", json_text(report), "soundleak-result.json", "application/json", width="stretch", on_click="ignore")
        st.download_button("Download heatmap", png_bytes(heatmap), "soundleak-heatmap.png", "image/png", width="stretch", on_click="ignore")
    if len(events) == 1:
        st.info("A single straight flight path leaves two mirrored listening positions with identical curves. "
                "The best grid point may be on either side. Combine a second flyover with a different heading to resolve this.")
    if result.best_x_m in (result.east_m[0], result.east_m[-1]) or result.best_y_m in (result.north_m[0], result.north_m[-1]):
        st.warning("The best fit lies at the search boundary. Expand or move the search area.")
    with st.expander("Inspect fitted tones and assumptions"):
        st.dataframe(report["events"], hide_index=True, width="stretch")
        st.write("The model fits one constant emitted frequency per event. It includes sound travel time, "
                 "but assumes still air, constant sound speed, a stationary recorder and local flat geometry. "
                 "Wind, changing engine RPM, echoes, track error and clock error can shift or invalidate the fit.")


with st.sidebar:
    st.markdown("### ◉ Soundleak")
    st.caption("AIRCRAFT DOPPLER LAB")
    mode = st.radio("Experiment", ["Soundleak Live", "Research workbench", "Public recording review", "Real-world replay", "Synthetic demo", "Analyze recording"], key="experiment_mode")
    st.divider()
    st.caption("Server-side analysis · no external inference API" if is_hosted() else "Local processing · no account needed")
    st.write("Match the changing pitch of a flyover to a known flight path.")

st.markdown('<div class="eyebrow">SOUND → MOTION → POSITION</div>', unsafe_allow_html=True)
st.title("Soundleak Live" if mode == "Soundleak Live" else "Soundleak research workbench" if mode == "Research workbench" else
         "Public recording review" if mode == "Public recording review" else "A location hidden in the sound.")
st.write("Hear the flyover. Follow the evidence. Watch the map update." if mode == "Soundleak Live" else
         "Review acoustic events, compare aircraft, and inspect competing locations." if mode == "Research workbench" else
         "Listen to each recording, inspect the extracted ridges, and save your judgments." if mode == "Public recording review" else
         "Trace an aircraft’s Doppler curve. Test where a listener would have heard it.")

if mode == "Soundleak Live":
    from soundleak.ui_live import render_live
    render_live()
elif mode == "Research workbench":
    from soundleak.ui_workbench import render_workbench
    render_workbench()
elif mode == "Public recording review":
    from soundleak.ui_public_review import render_public_review
    render_public_review()
elif mode == "Real-world replay":
    st.info("REAL AUDIO · Source-informed replay of two flyovers from colsto’s original one-hour recording.")
    st.caption("Aircraft identities, event windows, approximate region and clock prior come from a published reproduction. "
               "This replay tests our implementation on real inputs; it is not a blind location search.")
    required = [Path("data/real/reference")/name for name in
                ("original.webm", "trace_full_a1dfeb.json", "trace_full_ace391.json")]
    if not all(p.exists() for p in required):
        st.warning("The real recording and historical tracks have not finished downloading.")
        st.code(".venv/bin/python scripts/fetch_reference.py\n.venv/bin/python -m soundleak.cli real")
    else:
        try:
            with st.spinner("Matching real acoustic ridges to historical flight tracks…"):
                events, coarse, result, report = real_run()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Distance to public map marker", f"{report['distance_to_public_reference_m']:.0f} m")
            m2.metric("Frequency fit · RMSE", f"{result.best_rmse_hz:.2f} Hz")
            m3.metric("Real flyovers", str(len(events)))
            m4.metric("Refinement grid", f"{report['grid_spacing_m']:.0f} m")
            st.caption(f"Estimated listening position: {report['latitude']:.5f}°, {report['longitude']:.5f}°. "
                       "The public marker is approximate; distance to it is not measured microphone accuracy.")
            left, right = st.columns([1.4, 1], gap="large")
            with left:
                st.subheader("Real Doppler curves")
                st.caption("White: extracted from downloaded audio · Cyan: prediction at the fitted position")
                for index, (event, prediction) in enumerate(zip(events, result.predictions_hz), 1):
                    st.pyplot(event_figure(event, prediction), width="stretch")
                    clip = Path(f"outputs/real/reference/event-{index}.wav")
                    if clip.exists():
                        st.audio(clip.read_bytes(), format="audio/wav")
            with right:
                st.subheader("Where do both flyovers fit?")
                reference_xy = local_xy(*PUBLIC_REFERENCE, *ORIGIN)
                regional = heatmap_figure(coarse, events, reference_xy, reference_label="Public camera reference")
                st.pyplot(regional, width="stretch")
                with st.expander("Inspect the refined search"):
                    st.pyplot(heatmap_figure(result, events, reference_xy, reference_label="Public camera reference"), width="stretch")
                st.download_button("Download real-data report", json_text(report), "real-replay.json", "application/json", on_click="ignore", width="stretch")
                st.download_button("Download regional heatmap", png_bytes(regional), "real-heatmap.png", "image/png", on_click="ignore", width="stretch")
            with st.expander("Sources, timing and interpretation", expanded=True):
                st.markdown("Audio: [colsto · Unknown location – raw audio](https://www.youtube.com/watch?v=4CxVxuMVkMc). "
                            "Flights: [ADSB.lol · 8 September 2026 archive](https://github.com/adsblol/globe_history_2026/releases/tag/v2026.09.08-planes-readsb-staging-0). "
                            "Replay settings: [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST). "
                            "Comparison marker: [OspreyWatch · Oyster Bay](https://app.osprey-watch.org/nest/3916).")
                st.write("The recording-start clock prior was fitted in the published reproduction, rather than independently "
                         "measured at the recorder. This replay searches around that prior with one shared adjustment. "
                         "Manual event selection and known aircraft associations make this a regional replication. "
                         "A rounded public map marker does not establish survey-grade ground truth. "
                         "Wind, clock uncertainty, track interpolation and changing source tone remain limitations.")
                st.dataframe(report["events"], hide_index=True, width="stretch")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
elif mode == "Synthetic demo":
    st.info("SYNTHETIC EXPERIMENT · Generated audio and fictional flight paths. No real-world location claim.")
    with st.sidebar:
        use_two = st.toggle("Combine two flyovers", value=True)
        spacing = st.select_slider("Search grid spacing", options=[100, 150, 200, 300], value=150, format_func=lambda x: f"{x} m")
        st.caption("Turn off the second flyover to inspect the ambiguity from a single flight path.")
    with st.spinner("Extracting tones and testing candidate positions…"):
        events, result = demo_run(use_two, spacing)
    demo = demo_data()
    show_results(events, result, demo.origin, demo.truth_m[2], demo.truth_m)
    with st.expander("Listen to the experiment / download reproducible inputs"):
        st.audio(wav_bytes(demo), format="audio/wav")
        st.caption("Two moving tones mixed with broadband noise and a stationary 60 Hz tone. Fixed random seed: 7.")
        a, b = st.columns(2)
        a.download_button("Synthetic audio", wav_bytes(demo), "synthetic.wav", "audio/wav")
        b.download_button("Aircraft track CSV", tracks_csv(demo), "tracks.csv", "text/csv")
else:
    st.caption("Choose a tonal flyover, associate it with a flight track, and search a rough area. "
               "The first 180 seconds of each upload are analyzed in this app server's session.")
    audio_file = st.file_uploader("Audio or video", type=["wav", "mp3", "m4a", "mp4", "webm", "ogg", "flac", "mov"])
    track_file = st.file_uploader("Timestamped flight tracks · CSV", type=["csv"])
    with st.expander("CSV format and time alignment", expanded=not track_file):
        st.code("aircraft_id,time_s,latitude,longitude,altitude_m\nFLIGHT-A,-60,50.85,-1.49,1400\nFLIGHT-A,-58,50.85,-1.487,1400\nFLIGHT-A,-56,50.85,-1.484,1400", language="csv")
        st.write("Use actual sample times relative to audio sample zero: time_s = aircraft UTC timestamp − "
                 "recording-start UTC timestamp. Negative times are expected because sound takes time to arrive. "
                 "Provide dense track data spanning every selected event, with a lead-in of at least distance / 343 seconds. "
                 "Altitude is in meters above sea level; use the same vertical datum for the receiver.")
    a, b, c = st.columns(3)
    latitude = a.number_input("Search center · latitude", min_value=-80., max_value=80., value=50.85, format="%.5f")
    longitude = b.number_input("Search center · longitude", min_value=-179., max_value=179., value=-1.4, format="%.5f")
    altitude = c.number_input("Receiver elevation · m ASL", min_value=-500., max_value=8000., value=40.)
    if audio_file and track_file:
        try:
            tracks = read_tracks(track_file.getvalue().decode("utf-8-sig"), latitude, longitude)
            # A hash of file content lets Streamlit cache reruns without retaining disk uploads.
            @st.cache_data(max_entries=3)
            def uploaded_audio(contents, suffix):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / ("recording" + suffix)
                    path.write_bytes(contents)
                    return decode_audio(path)
            audio, sr = uploaded_audio(audio_file.getvalue(), Path(audio_file.name).suffix)
            duration = len(audio) / sr
            st.caption(f"Decoded {duration:.1f} seconds · {len(tracks)} aircraft tracks")
            with st.form("analysis"):
                count = st.selectbox("Number of selected flyovers", [1, 2, 3])
                # Keep three rows visible so changing the count inside the form is unambiguous.
                windows = []
                for i in range(3):
                    st.markdown(f"**Event {i+1}**" + (" · used when selected above" if i else ""))
                    cols = st.columns([1.3, 1, 1, 1, 1])
                    name = cols[0].selectbox("Aircraft", list(tracks), key=f"aircraft-{i}")
                    start = cols[1].number_input("Start · s", value=0., min_value=0., max_value=duration-4, key=f"start-{i}")
                    end = cols[2].number_input("End · s", value=min(60., duration), min_value=4., max_value=duration, key=f"end-{i}")
                    low = cols[3].number_input("Low · Hz", min_value=1., max_value=1999., value=80., key=f"low-{i}")
                    high = cols[4].number_input("High · Hz", min_value=2., max_value=2000., value=180., key=f"high-{i}")
                    windows.append((name, start, end, low, high))
                a, b, c = st.columns(3)
                radius = a.slider("Search half-width · km", 1, 15, 3)
                grid_n = b.select_slider("Grid points per axis", [21, 31, 41, 61], value=31)
                sound_speed = c.number_input("Sound speed · m/s", value=343., min_value=300., max_value=370.)
                offset_limit = st.select_slider("Shared clock offset search · seconds", [0, 2, 5, 10], value=0)
                submitted = st.form_submit_button("Extract ridges & search", type="primary")
            if submitted:
                events = [Event(tracks[n], extract_ridge(audio, sr, start, end, low, high)) for n, start, end, low, high in windows[:count]]
                for event in events:
                    if not event.ridge.usable:
                        st.pyplot(event_figure(event))
                grid = np.linspace(-radius * 1000, radius * 1000, grid_n)
                progress = st.progress(0., text="Testing listening positions…")
                offsets = np.arange(-offset_limit, offset_limit+1) if offset_limit else [0]
                result = search(events, grid, grid, altitude, sound_speed, offsets, progress=lambda x: progress.progress(x))
                progress.empty()
                show_results(events, result, (latitude, longitude), altitude)
        except (ValueError, UnicodeError) as exc:
            st.error(str(exc))

st.divider()
st.caption("Research prototype · Automatic acoustic proposals with manual review · Finite regional fits, not calibrated location probabilities")
