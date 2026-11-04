"""Reproducible acoustic screening of public aircraft passages.

Pathak's 40-passage release is an acoustic component check. No author-picked
ridges, aircraft trajectories, UTC priors or microphone positions enter discovery.
It is not an independently timed/georeferenced receiver-localization pilot.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import md5, sha256
import json
from pathlib import Path, PurePosixPath
import tempfile
import time
import wave
import zipfile

import requests

from .archive import file_sha256
from .audio import decode_audio
from .discovery import DiscoveryConfig, discover

RECORD_ID = 23424278
SOURCE = f"https://zenodo.org/records/{RECORD_ID}"
ARCHIVES = (
    "Raw data_V_constant_H_constant.zip",
    "Raw data_V_constant_H_varying.zip",
    "Raw data_V_varying_H_constant.zip",
    "Raw data_V_varying_H_varying.zip",
)


def asset_matches(path, asset):
    path = Path(path)
    if not path.exists() or path.stat().st_size != asset["size"]:
        return False
    digest = md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return asset["checksum"] == "md5:"+digest.hexdigest()


def download_asset(asset, target, chunk_size=1024*1024):
    """Resume bounded ranges; publish only after the complete digest agrees."""
    target = Path(target)
    if asset_matches(target, asset):
        return
    ranges = target.with_suffix(target.suffix+".ranges")
    ranges.mkdir(exist_ok=True)

    def fetch(start):
        length = min(chunk_size, asset["size"]-start)
        chunk = ranges/f"{start:012d}"
        if chunk.exists() and chunk.stat().st_size == length:
            return chunk
        end = start+length-1
        url = asset["links"]["self"]
        # Distinguish cache entries on intermediaries that do not vary on Range.
        url += ("&" if "?" in url else "?")+f"soundleak_range={start}-{end}"
        for attempt in range(3):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{asset['size']}":
                        raise ValueError("Public server did not honor the requested byte range.")
                    temporary = chunk.with_suffix(".part")
                    size = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(64*1024):
                            size += len(block)
                            if size > length:
                                raise ValueError("Public range exceeded its declared length.")
                            stream.write(block)
                if size != length:
                    raise ValueError("Incomplete public archive range.")
                temporary.replace(chunk)
                return chunk
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(4) as pool:
        chunks = list(pool.map(fetch, range(0, asset["size"], chunk_size)))
    temporary = target.with_suffix(target.suffix+".part")
    with temporary.open("wb") as stream:
        for chunk in chunks:
            stream.write(chunk.read_bytes())
    if not asset_matches(temporary, asset):
        raise ValueError("Public archive failed its published checksum; ranges retained for inspection.")
    temporary.replace(target)
    for chunk in chunks:
        chunk.unlink()
    if not list(ranges.iterdir()):
        ranges.rmdir()


def fetch_public_aircraft(directory, progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory/"record.json"
    if metadata.exists():
        record = json.loads(metadata.read_text())
    else:
        response = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=30)
        response.raise_for_status()
        record = response.json()
        metadata.write_text(json.dumps(record, indent=2))
    if int(record["id"]) != RECORD_ID:
        raise ValueError("Unexpected public dataset record.")
    assets = {a["key"]: a for a in record["files"]}

    def download(name):
        asset, target = assets[name], directory/name
        if asset["size"] > 100_000_000:
            raise ValueError("Public archive exceeds expected download size.")
        download_asset(asset, target)
        if progress:
            progress(name)
        return {"file": name, "bytes": asset["size"], "published_checksum": asset["checksum"],
                "sha256": file_sha256(target)}

    with ThreadPoolExecutor(2) as pool:
        downloaded = list(pool.map(download, ARCHIVES))
    # Validate the complete cohort before any discovery results are calculated.
    clips = []
    identities = set()
    for archive in downloaded:
        with zipfile.ZipFile(directory/archive["file"]) as source:
            for member in sorted(source.infolist(), key=lambda m: m.filename):
                name = PurePosixPath(member.filename)
                if "__MACOSX" in name.parts or name.name != "audio.wav":
                    continue
                if len(name.parts) != 3 or member.file_size > 100_000_000:
                    raise ValueError("Unexpected public audio member.")
                identity = name.parent.name
                if identity in identities:
                    raise ValueError("Duplicate public passage identity.")
                identities.add(identity)
                clips.append({"id": identity, "category": name.parts[0], "archive": archive["file"],
                              "member": member.filename, "uncompressed_bytes": member.file_size})
    if len(clips) != 40:
        raise ValueError(f"Expected all 40 public passages; found {len(clips)}.")
    manifest = {"schema_version": 1, "source": SOURCE, "creator": "Praveen Pathak",
                "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "record_metadata_sha256": file_sha256(metadata), "archives": downloaded, "clips": clips,
                "selection": "Every raw audio.wav from all four release archives, sorted by archive and member name; no selection by detector output.",
                "independent_localization_eligible": False,
                "localization_limits": ["Independent receiver coordinates have not been established.",
                    "UTC-stamped extracted ridge data does not establish the UTC time of WAV sample zero.",
                    "No flight catalog or reference coordinates enter this acoustic screen."]}
    (directory/"screen-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_public_screen(directory, output, config=DiscoveryConfig(), progress=None):
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory/"screen-manifest.json").read_text())
    if len(manifest["clips"]) != 40 or len({c["id"] for c in manifest["clips"]}) != 40:
        raise ValueError("The public acoustic screen requires all 40 distinct passages.")
    for asset in manifest["archives"]:
        if file_sha256(directory/asset["file"]) != asset["sha256"]:
            raise ValueError("Public archive failed its input-integrity check.")
    # This record is written before discovery, so the declared cohort/configuration
    # can be inspected independently of the outcome. Never silently omit failures.
    output.parent.mkdir(parents=True, exist_ok=True)
    declaration = {"manifest_sha256": file_sha256(directory/"screen-manifest.json"),
                   "discovery_config": asdict(config), "case_ids": [c["id"] for c in manifest["clips"]],
                   "analysis_code_sha256": {n: file_sha256(Path(__file__).with_name(n))
                                            for n in ("public_data.py", "audio.py", "discovery.py")}}
    output.with_suffix(".declaration.json").write_text(json.dumps(declaration, indent=2))
    cases = []
    for clip in manifest["clips"]:
        row = dict(clip)
        try:
            with zipfile.ZipFile(directory/clip["archive"]) as source:
                # ZipFile.read validates the entire member CRC, unlike partial reads.
                contents = source.read(clip["member"])
            if len(contents) != clip["uncompressed_bytes"]:
                raise ValueError("Public audio member has unexpected length.")
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"recording.wav"
                path.write_bytes(contents)
                with wave.open(str(path), "rb") as wav:
                    original_duration = wav.getnframes()/wav.getframerate()
                if original_duration > 600:
                    raise ValueError("Unexpected recording longer than ten minutes; refusing silent truncation.")
                audio, sr = decode_audio(path, max_duration_s=600)
            result = discover(audio, sr, config)
            row.update({"state": "screened", "duration_s": result.duration_s,
                        "source_duration_s": original_duration, "audio_sha256": sha256(contents).hexdigest(),
                        "full_member_crc_verified": True, "proposal_count": len(result.proposals),
                        "below_minimum_event_duration": result.duration_s < config.min_duration_s,
                        "discovery": result.metadata()})
        except (ValueError, OSError, EOFError, RuntimeError, zipfile.BadZipFile, wave.Error) as exc:
            row.update({"state": "failed", "reason": str(exc)})
        cases.append(row)
        if progress:
            progress(len(cases), len(manifest["clips"]))
    screened = [c for c in cases if c["state"] == "screened"]
    found = sum(c["proposal_count"] > 0 for c in screened)
    report = {"schema_version": 1, "source": manifest["source"], "creator": manifest["creator"],
              "license": manifest["license"], "selection": manifest["selection"], **declaration,
              "declared_cases": len(cases), "screened_cases": len(screened), "failed_cases": len(cases)-len(screened),
              "cases_with_proposals": found, "cases_without_proposals": len(screened)-found,
              "total_acoustic_proposals": sum(c["proposal_count"] for c in screened),
              "total_audio_seconds": sum(c["duration_s"] for c in screened),
              "independent_localization_eligible": False, "localization_limits": manifest["localization_limits"],
              "cases": cases,
              "interpretation": "Acoustic component check on published aircraft passages using frozen discovery defaults. Proposed patterns are unconfirmed; clip-level proposal presence is not aircraft-identification accuracy, localization accuracy or calibrated recall. No successful location claim is made; all misses and failures are retained."}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
