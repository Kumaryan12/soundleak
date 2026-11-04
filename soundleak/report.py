import io
import json
import numpy as np
from matplotlib.figure import Figure

from .physics import latlon

BG = "#111b29"
TEXT = "#c6d3e3"
CYAN = "#51e0d2"


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#2a3b50")


def event_figure(event, prediction=None):
    return ridge_figure(event.ridge, event.track.name, prediction)


def ridge_figure(ridge, title="Acoustic proposal", prediction=None, prediction_label="Predicted at best fit"):
    fig = Figure(figsize=(9, 3.4), facecolor=BG, layout="constrained")
    ax = fig.subplots()
    ax.pcolormesh(ridge.time_s, ridge.frequency_bins_hz, ridge.power_db, shading="auto", cmap="magma",
                  vmin=np.percentile(ridge.power_db, 15), vmax=np.percentile(ridge.power_db, 99.7), rasterized=True)
    ax.plot(ridge.time_s, ridge.frequency_hz, color="white", lw=1, alpha=.65, label="Measured ridge")
    if prediction is not None:
        ax.plot(ridge.time_s, prediction, color=CYAN, lw=1.8, label=prediction_label)
    ax.set(xlabel="Recording time · seconds", ylabel="Frequency · Hz", title=title)
    style(ax)
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, loc="upper right")
    return fig


def timeline_figure(audio, sample_rate, proposals, selected_id=None):
    fig = Figure(figsize=(12, 2.6), facecolor=BG, layout="constrained")
    ax = fig.subplots()
    seconds = len(audio)//sample_rate
    if seconds:
        rms_db = 10*np.log10(np.maximum(np.mean(np.asarray(audio[:seconds*sample_rate]).reshape(seconds, sample_rate)**2, axis=1), 1e-20))
        ax.plot(np.arange(seconds)/60, rms_db, color=TEXT, lw=.7, alpha=.7)
        for p in proposals:
            a, b = p.ridge.time_s[[0, -1]]/60
            ax.axvspan(a, b, color=CYAN, alpha=.45 if p.id == selected_id else .08)
        ax.set(xlabel="Recording time · minutes", ylabel="Audio level · dBFS", title="Whole-recording scan · shaded intervals are acoustic proposals")
    style(ax)
    return fig


def review_spectrogram_figure(times, frequencies, power_db, proposals, selected_id=None,
                              time_range=None, frequency_range=None):
    """Display the recording context and original proposal ridges without retracing."""
    fig = Figure(figsize=(12, 4.4), facecolor=BG, layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(power_db, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
                   extent=(times[0]-.125, times[-1]+.125, frequencies[0]-.125, frequencies[-1]+.125),
                   vmin=np.percentile(power_db, 15), vmax=np.percentile(power_db, 99.7))
    for i, p in enumerate(proposals, 1):
        selected = p.id == selected_id
        ax.plot(p.ridge.time_s, p.ridge.frequency_hz, color=CYAN if selected else "white",
                lw=2.2 if selected else 1.2, alpha=1 if selected else .8)
        middle = len(p.ridge.time_s)//2
        ax.annotate(f"P{i}", (p.ridge.time_s[middle], p.ridge.frequency_hz[middle]),
                    xytext=(3, 8), textcoords="offset points", color=CYAN if selected else "white",
                    fontsize=9, bbox={"facecolor": BG, "alpha": .8, "edgecolor": "none", "pad": 2})
    ax.set(xlabel="Recording time · seconds", ylabel="Frequency · Hz",
           title="Recording spectrogram · extracted proposal ridges")
    if time_range:
        ax.set_xlim(*time_range)
    if frequency_range:
        ax.set_ylim(*frequency_range)
    style(ax)
    cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.02)
    cb.set_label("Spectral level · dB (relative)", color=TEXT, fontsize=9)
    cb.ax.tick_params(colors=TEXT, labelsize=8)
    return fig


def association_figure(result, tracks, hypothesis=None):
    fig = Figure(figsize=(6, 5.1), facecolor=BG, layout="constrained")
    ax = fig.subplots()
    im = ax.pcolormesh(result.east_m/1000, result.north_m/1000, result.surface_hz, shading="auto", cmap="viridis_r")
    for t in tracks.values():
        ax.plot(t.xyz_m[:, 0]/1000, t.xyz_m[:, 1]/1000, color="white", lw=.8, alpha=.3)
    for i, h in enumerate(result.hypotheses):
        ax.scatter(h.east_m/1000, h.north_m/1000, color=CYAN, s=35, marker="o")
        ax.annotate(str(i+1), (h.east_m/1000, h.north_m/1000), xytext=(5, 5), textcoords="offset points", color=TEXT)
    if hypothesis is not None:
        ax.scatter(hypothesis.east_m/1000, hypothesis.north_m/1000, marker="+", color=CYAN, s=180, lw=2)
    ax.set(xlim=(result.east_m[0]/1000, result.east_m[-1]/1000), ylim=(result.north_m[0]/1000, result.north_m[-1]/1000),
           xlabel="East of search center · km", ylabel="North of search center · km", title="Ranked regional hypotheses")
    ax.set_aspect("equal")
    style(ax)
    cb = fig.colorbar(im, ax=ax, shrink=.8)
    cb.set_label("Capped event objective · Hz · lower is better", color=TEXT, fontsize=9)
    cb.ax.tick_params(colors=TEXT)
    return fig


def heatmap_figure(result, events, truth=None, reference_label="Synthetic ground truth"):
    fig = Figure(figsize=(6, 5.8), facecolor=BG, layout="constrained")
    ax = fig.subplots()
    # Display an actual residual, with no conversion to an uncalibrated probability.
    surface = np.ma.masked_invalid(result.rmse_hz)
    im = ax.pcolormesh(result.east_m / 1000, result.north_m / 1000, surface, shading="auto", cmap="viridis_r")
    for event in events:
        p = event.track.xyz_m
        ax.plot(p[:, 0] / 1000, p[:, 1] / 1000, color="white", alpha=.4, lw=1, ls="--")
    ax.scatter(result.best_x_m / 1000, result.best_y_m / 1000, marker="+", s=190, color=CYAN, lw=2.5, label="Best grid fit")
    if truth is not None:
        ax.scatter(truth[0] / 1000, truth[1] / 1000, marker="*", s=110, color="#ffbe76", label=reference_label)
    ax.set(xlim=(result.east_m[0]/1000, result.east_m[-1]/1000), ylim=(result.north_m[0]/1000, result.north_m[-1]/1000),
           xlabel="East of search center · km", ylabel="North of search center · km", title="Where does the curve fit?")
    ax.set_aspect("equal")
    style(ax)
    colorbar = fig.colorbar(im, ax=ax, shrink=.78)
    colorbar.set_label("Frequency RMSE · Hz · lower is better", color=TEXT, fontsize=9)
    colorbar.ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, loc="upper left")
    return fig


def summary(result, events, origin, altitude_m, truth=None):
    latitude, longitude = latlon(result.best_x_m, result.best_y_m, *origin)
    report = {
        "mode": "synthetic" if truth is not None else "uploaded_recording",
        "latitude": float(latitude), "longitude": float(longitude),
        "east_m": result.best_x_m, "north_m": result.best_y_m,
        "assumed_receiver_altitude_m": float(altitude_m),
        "assumed_sound_speed_mps": result.sound_speed_mps,
        "frequency_rmse_hz": result.best_rmse_hz,
        "clock_offset_added_to_audio_s": result.clock_offset_s,
        "grid_spacing_m": float(result.east_m[1] - result.east_m[0]),
        "search_east_extent_m": [float(result.east_m[0]), float(result.east_m[-1])],
        "search_north_extent_m": [float(result.north_m[0]), float(result.north_m[-1])],
        "events": [{"aircraft_id": event.track.name, "fitted_source_tone_hz": tone,
                    "ridge_start_s": float(event.ridge.time_s[0]),
                    "ridge_end_s": float(event.ridge.time_s[-1]),
                    "ridge_prominence_db": float(np.median(event.ridge.prominence_db))}
                   for event, tone in zip(events, result.source_frequencies_hz)],
        "limitations": ["Fit error is not a location probability.",
                        "Constant emitted tone per event, still air, flat local geometry, known receiver elevation.",
                        "Manual time window, frequency band and aircraft association.",
                        "Synthetic validation does not establish real-world accuracy."],
    }
    if truth is not None:
        report["synthetic_position_error_m"] = float(np.hypot(result.best_x_m-truth[0], result.best_y_m-truth[1]))
    return report


def png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor=fig.get_facecolor())
    return buffer.getvalue()


def json_text(report):
    return json.dumps(report, indent=2, allow_nan=False)

