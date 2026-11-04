"""Real annotated background controls, selected before detector evaluation.

AeroSonicDB v1.1.2, Blake Downward, CC BY-NC 4.0. One contiguous 90-second
excerpt per environmental recording, from the earliest 100-second run labeled
no-aircraft, with five-second margins. No stitching or repeated short clips.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import wave

import requests

from .archive import file_sha256
from .audio import decode_audio
from .discovery import DiscoveryConfig, discover

RECORD_URL = "https://zenodo.org/api/records/10215080"


def background_windows(rows, minimum_bins=20):
    windows = {}
    for column in rows[0]:
        start = None
        for i, row in enumerate(rows):
            if row[column] == "0":
                if start is None:
                    start = i
                if i-start+1 >= minimum_bins:
                    windows[int(column)+1] = ((start+1)*5, (start+minimum_bins-1)*5)
                    break
            else:
                start = None
    return windows


def fetch_background_controls(directory, progress=None):
    from remotezip import RemoteZip
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(RECORD_URL, timeout=30)
    response.raise_for_status()
    record = response.json()
    (directory/"record.json").write_text(json.dumps(record, indent=2))
    assets = {f["key"]: f for f in record["files"]}
    labels = assets["environment_class_mappings.csv"]
    response = requests.get(labels["links"]["self"], timeout=30)
    response.raise_for_status()
    if "md5:"+hashlib.md5(response.content).hexdigest() != labels["checksum"]:
        raise ValueError("Background-label checksum failed.")
    (directory/"environment_class_mappings.csv").write_bytes(response.content)
    rows = list(csv.DictReader(response.content.decode().splitlines()))
    windows = background_windows(rows)
    if len(windows) != 6:
        raise ValueError("Not all six environmental recordings have the required labeled background span.")
    url = "https://zenodo.org/records/10215080/files/env_audio.zip?download=1"

    def fetch(item):
        number, (start, end) = item
        member = f"env_audio/{number}_AUDIO.wav"
        output = directory/f"background-{number}.wav"
        with RemoteZip(url, timeout=60) as archive:
            info = archive.getinfo(member)
            with archive.open(member) as source:
                with wave.open(source, "rb") as wav:
                    sample_rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                    wav.setpos(start*sample_rate)
                    pcm = wav.readframes((end-start)*sample_rate)
            if len(pcm) != (end-start)*sample_rate*channels*width:
                raise ValueError("Incomplete background excerpt.")
            with wave.open(str(output), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(width)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm)
        if progress:
            progress(number)
        return {"id": f"aerosonic-background-{number}", "file": output.name, "source_member": member,
                "mapping_column": str(number-1), "start_s": start, "end_s": end, "duration_s": end-start,
                "publisher_label": "0 (no aircraft)", "sample_rate": sample_rate,
                "sha256": file_sha256(output), "source_member_crc32_expected": f"{info.CRC:08x}",
                "full_member_crc_verified": False,
                "integrity_note": "HTTPS partial-member excerpt; full ZIP/member not downloaded. Exported excerpt SHA-256 retained."}
    with ThreadPoolExecutor(3) as pool:
        clips = list(pool.map(fetch, sorted(windows.items())))
    manifest = {"source": "https://zenodo.org/records/10215080", "creator": "Blake Downward",
                "license": "CC BY-NC 4.0", "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "selection": "Earliest 100-second contiguous no-aircraft annotation run per recording; 90-second excerpt with 5-second margins, chosen before scanning.",
                "annotation_checksum": labels["checksum"], "archive_checksum_expected": assets["env_audio.zip"]["checksum"],
                "mapping_evidence": "https://github.com/aerosonicdb/AeroSonicDB-YPAD0523/blob/main/aerosonicdb/features/extract_env_mfccs.py",
                "clips": clips}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_background_controls(directory, output, config=DiscoveryConfig()):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    cases = []
    for clip in manifest["clips"]:
        path = directory/clip["file"]
        if file_sha256(path) != clip["sha256"]:
            raise ValueError("Background control failed its input-integrity check.")
        audio, sr = decode_audio(path, max_duration_s=clip["duration_s"])
        result = discover(audio, sr, config)
        cases.append({"id": clip["id"], "duration_s": result.duration_s,
                      "proposal_count": len(result.proposals), "discovery": result.metadata(),
                      "source_excerpt_sha256": clip["sha256"]})
    report = {"source": manifest["source"], "license": manifest["license"], "selection": manifest["selection"],
              "cases": cases, "total_background_seconds": sum(c["duration_s"] for c in cases),
              "clips_with_proposals": sum(c["proposal_count"] > 0 for c in cases),
              "total_acoustic_proposals": sum(c["proposal_count"] for c in cases),
              "interpretation": "Exploratory acoustic rejection check on six correlated environmental-source excerpts, not a calibrated false-location rate or a blind geolocation benchmark."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
