# Wi-Fi link experiment

Soundleak's Wi-Fi lab analyzes **RSSI**, the signal strength reported by a Mac's
current Wi-Fi connection. It does not infer human presence, count people, identify
anyone, estimate pose or locate a person. RSSI is an aggregate link measurement;
CSI-based research uses richer channel measurements that this collector does not
provide. This experiment has not been validated as a human-activity detector.

## Collect with a Mac and router

Keep `capture_wifi.py` and `capture_wifi.swift` in the same folder. Connect the Mac
to your router and keep both devices stationary. The tool requires macOS and
Apple Swift command-line tools. From Terminal in that folder:

```sh
python3 capture_wifi.py --duration 60 --output wifi-capture.csv
```

In the project checkout the equivalent command is:

```sh
.venv/bin/python scripts/capture_wifi.py --duration 60 --output outputs/wifi/experiment-1.csv
```

The first run may take extra time to compile Swift. Collection lasts 5–300 seconds
and defaults to two API queries per second. The Wi-Fi driver may update its RSSI
less frequently; repeated readings are not independent measurements. The tool
does not scan networks or record SSIDs, BSSIDs, credentials or packet contents.
It only reads RSSI, aggregate noise, channel number, connection availability and
elapsed time. It refuses to overwrite an existing capture.

Use the first 10 seconds as a representative static reference. After that,
optionally walk through the area while leaving the devices in place. Record an
independent activity log if you want to evaluate any association. Also collect
controls: an unchanged room, normal network traffic and other environmental
changes. Do not assume a signal change proves that a person caused it.

## View results

Open https://soundleak-live.streamlit.app/ and select **Wi-Fi lab** in the sidebar.
Upload the CSV, set the reference duration and confirm how the reference was
recorded. The website runs in the cloud and cannot directly read the Mac's Wi-Fi
measurements. Uploading a capture sends those numeric measurements to the app
server for analysis. Captures are not added to the public GitHub repository.

When running the project locally on macOS, the lab also offers a button for a
30-second capture. The separately selected synthetic demonstration is generated
data, not a real sensing result.

## Interpretation

The lab computes the standard deviation of RSSI in trailing three-second windows.
It compares each window with the median reference variability plus the greater of
2 dB or six scaled median absolute deviations. Exceedances must persist for at
least one second to form a reported interval. This is an exploratory threshold,
not a calibrated probability or trained activity classifier.

Missing measurements and timing gaps invalidate affected windows. Reference
periods with inadequate coverage and captures crossing known Wi-Fi channels are
rejected. An unchanged channel does not rule out roaming or other link changes.
Traffic, interference, transmit power, moving the Mac, pets and environmental
changes can all confound the measurements. Quiet RSSI does not establish absence.

CSV columns are `time_s,rssi_dbm,noise_dbm,channel,status`. Noise and channel are
optional; missing RSSI uses blank values and `status=unavailable`. Connected RSSI
must be negative. Sample times must increase, with at least 12 samples and enough
duration for reference and subsequent comparison. Limits: 2 MB, 10,000 rows and
20 minutes. JSON exports contain numeric diagnostics and explicitly retain
`human_presence_status=not_estimated` and `location_status=not_estimated`.

## Toward CSI sensing

Person detection would need suitable channel measurements, representative labeled
data, calibration and independent tests of false alarms and missed detections.
The current Mac collector exposes RSSI through Apple's public CoreWLAN API; it
does not expose raw CSI. Supported ESP32 hardware running suitable firmware is a
possible future acquisition path. A CSI pipeline is not included in this release.

References:

- [Apple CoreWLAN RSSI](https://developer.apple.com/documentation/corewlan/cwinterface/rssivalue())
- [Apple CoreWLAN noise](https://developer.apple.com/documentation/corewlan/cwinterface/noisemeasurement())
- [Espressif CSI toolkit](https://github.com/espressif/esp-csi)
