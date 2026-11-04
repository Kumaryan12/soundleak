# Hosting Soundleak

Prepare a deployment checkout containing the app, pinned model, reference replay,
regional flight catalog, public review assets and original public recordings:

```sh
.venv/bin/python scripts/prepare_deployment.py --output outputs/deployment/source
```

The output directory must not already exist. The allowlist omits credentials,
virtual environments, private review notes, user sessions and large unused daily
flight archives. `deploy/bundle-manifest.json` records each included file's hash.

## Streamlit Community Cloud

Push the prepared checkout to a GitHub repository. In Community Cloud, choose that
repository, branch `main`, entrypoint `streamlit_app.py`, and Python **3.12** in
Advanced settings. `requirements.txt` includes the pinned runtime dependencies;
`packages.txt` installs FFmpeg. No API keys are required for the bundled demo or
sound classification. A private repository keeps the source private; application
visibility is configured separately in Streamlit's sharing settings.

The dedicated entrypoint enables hosted mode before loading the application.
Microphone permission requires HTTPS. Recorded and uploaded audio is processed on
the app server. Hosted notes and newly created cases use separate temporary
directories for each browser session. Download results before closing the session
or restarting the app. Shared public examples are read-only application inputs.

Community Cloud is suitable for a small demonstration. The replay is precomputed,
and the sound classifier does not estimate a recording's location. Arbitrary
recordings require separately supplied timestamped aircraft tracks for Doppler
matching. Concurrent large research jobs can exceed a small host's resources.

## Docker

Build from the prepared checkout, not the full research directory:

```sh
docker build -t soundleak outputs/deployment/source
docker run --rm -p 8501:8501 soundleak
```

The image runs as an unprivileged user, accepts a host-provided `PORT`, enables
hosted mode, and provides `/_stcore/health`. Use a single instance or session-aware
WebSocket routing. Start with at least 1 GB RAM and measure under the intended
workload before enabling public research jobs. Container files are temporary;
this deployment does not provide persistent accounts, uploads or annotations.
