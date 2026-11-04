"""Resumable, checksum-verified ADSB.lol daily archive acquisition.

Downloads public release assets in bounded ranges and never executes upstream
code. Regional selection uses reported time and position only, not audio events
or aircraft identities. Downloaded records retain ADSB.lol ODbL attribution.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import tarfile
import threading
import time

import numpy as np
import requests

from .physics import local_xy


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_daily_archive(day, directory, workers=12, chunk_size=16*1024*1024, progress=None):
    day = datetime.strptime(day, "%Y-%m-%d").date().isoformat()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tag = f"v{day.replace('-', '.')}-planes-readsb-staging-0"
    repo = f"https://api.github.com/repos/adsblol/globe_history_{day[:4]}"
    metadata_path = directory/"release.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("tag_name") != tag:
            raise ValueError("Cached release metadata is for a different day.")
    else:
        response = requests.get(f"{repo}/releases/tags/{tag}", timeout=30)
        response.raise_for_status()
        metadata = response.json()
        metadata_path.write_text(json.dumps(metadata, indent=2))
    assets = sorted([a for a in metadata["assets"] if a["name"].startswith(tag+".tar.")], key=lambda a: a["name"])
    if not assets or [a["name"].rsplit(".", 1)[-1] for a in assets] != ["aa", "ab"]:
        raise ValueError("Expected the two documented tar archive parts aa and ab.")
    completed, jobs, urls = [], [], {}
    total = sum(a["size"] for a in assets)
    done, lock = 0, threading.Lock()
    for asset in assets:
        name = asset["name"]
        if Path(name).name != name or not str(asset.get("digest", "")).startswith("sha256:"):
            raise ValueError("Archive asset lacks a safe name or published SHA-256 digest.")
        destination = directory/name
        completed.append(destination)
        if destination.exists() and destination.stat().st_size == asset["size"] and file_sha256(destination) == asset["digest"].split(":", 1)[1]:
            done += asset["size"]
            continue
        response = requests.get(asset["browser_download_url"], allow_redirects=False, timeout=30)
        response.raise_for_status()
        urls[name] = response.headers.get("Location", asset["browser_download_url"])
        chunks = directory/(name+".chunks")
        chunks.mkdir(exist_ok=True)
        for start in range(0, asset["size"], chunk_size):
            length = min(chunk_size, asset["size"]-start)
            path = chunks/f"{start:012d}.part"
            if path.exists() and path.stat().st_size == length:
                done += length
            else:
                jobs.append((asset, start, length, path))

    def download(job):
        nonlocal done
        asset, start, length, path = job
        temporary = path.with_suffix(".partial")
        for attempt in range(4):
            try:
                with requests.get(urls[asset["name"]], headers={"Range": f"bytes={start}-{start+length-1}"},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{start+length-1}/{asset['size']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Archive server did not honor the requested byte range.")
                    with temporary.open("wb") as output:
                        for block in response.iter_content(1024*1024):
                            output.write(block)
                if temporary.stat().st_size != length:
                    raise ValueError("Incomplete archive range download.")
                temporary.replace(path)
                with lock:
                    done += length
                    if progress:
                        progress(done, total)
                return
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 8))

    if progress:
        progress(done, total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(download, job) for job in jobs]):
            future.result()
    for asset, destination in zip(assets, completed):
        if destination.exists() and destination.stat().st_size == asset["size"] and file_sha256(destination) == asset["digest"].split(":", 1)[1]:
            continue
        temporary = destination.with_suffix(destination.suffix+".partial")
        with temporary.open("wb") as output:
            for start in range(0, asset["size"], chunk_size):
                with (directory/(asset["name"]+".chunks")/f"{start:012d}.part").open("rb") as chunk:
                    for block in iter(lambda: chunk.read(4*1024*1024), b""):
                        output.write(block)
        if temporary.stat().st_size != asset["size"] or file_sha256(temporary) != asset["digest"].split(":", 1)[1]:
            raise ValueError("Archive checksum failed; cached ranges are retained for inspection.")
        temporary.replace(destination)
        # Remove only this downloader's completed temporary ranges after verification.
        chunks = directory/(asset["name"]+".chunks")
        for chunk in chunks.glob("*.part"):
            chunk.unlink()
        if not list(chunks.iterdir()):
            chunks.rmdir()
    return completed, metadata


class ConcatenatedReader(io.RawIOBase):
    """Read split tar parts in order without creating another multi-GB copy."""
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = Path(path).open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def decode_trace(contents, limit_bytes=50_000_000):
    if contents[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(contents)) as stream:
            contents = stream.read(limit_bytes+1)
    if len(contents) > limit_bytes:
        raise ValueError("Trace exceeds the documented decompression limit.")
    result = json.loads(contents)
    if not isinstance(result, dict) or not isinstance(result.get("trace"), list):
        raise ValueError("Invalid native trace record.")
    return result


def trace_intersects_region(data, origin, start_unix_s, end_unix_s, radius_m):
    base = float(data["timestamp"])
    for row in data["trace"]:
        if len(row) < 4 or row[1] is None or row[2] is None or row[3] == "ground":
            continue
        if not start_unix_s <= base+float(row[0]) <= end_unix_s:
            continue
        x, y = local_xy(float(row[1]), float(row[2]), *origin)
        if np.isfinite(x) and np.isfinite(y) and np.hypot(x, y) <= radius_m:
            return True
    return False


def extract_region(parts, output, origin, recording_start_unix_s, duration_s, radius_m=75_000.,
                   lead_s=200., trail_s=60., release_metadata=None, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    traces = output/"traces"
    traces.mkdir()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_release": (release_metadata or {}).get("html_url"), "license": "ODbL-1.0",
                "origin_latitude_longitude": list(origin), "radius_m": radius_m,
                "time_start_unix_s": recording_start_unix_s-lead_s,
                "time_end_unix_s": recording_start_unix_s+duration_s+trail_s,
                "selection": "At least one reported airborne position within the time interval and horizontal radius; no aircraft-type or audio-event filtering.",
                "archive_parts": [{"file": Path(p).name, "bytes": Path(p).stat().st_size, "sha256": file_sha256(p)} for p in parts],
                "trace_members_scanned": 0, "selected_aircraft": [], "errors": []}
    with io.BufferedReader(ConcatenatedReader(parts), buffer_size=1024*1024) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                name = member.name.lstrip("./")
                if not member.isfile() or not name.startswith("traces/") or not Path(name).name.startswith("trace_full_"):
                    continue
                manifest["trace_members_scanned"] += 1
                try:
                    if member.size > 50_000_000:
                        raise ValueError("Compressed trace exceeds size limit.")
                    raw = archive.extractfile(member).read()
                    data = decode_trace(raw)
                    if trace_intersects_region(data, origin, manifest["time_start_unix_s"], manifest["time_end_unix_s"], radius_m):
                        identity = str(data["icao"])
                        if not identity or any(c not in "0123456789abcdef~" for c in identity):
                            raise ValueError("Invalid aircraft identifier.")
                        target = traces/(identity+".json.gz")
                        if target.exists():
                            raise ValueError("Duplicate aircraft trace in daily archive.")
                        target.write_bytes(raw if raw[:2] == b"\x1f\x8b" else gzip.compress(raw, mtime=0))
                        manifest["selected_aircraft"].append({"icao": identity, "type": data.get("t"),
                            "registration": data.get("r"), "member": member.name,
                            "member_data_offset": member.offset_data, "file": str(target.relative_to(output)),
                            "sha256": file_sha256(target), "decoded_sha256": sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    manifest["errors"].append({"member": member.name, "error": str(exc)})
                if progress and manifest["trace_members_scanned"] % 2000 == 0:
                    progress(manifest["trace_members_scanned"], len(manifest["selected_aircraft"]))
    manifest["complete_archive_scan"] = True
    manifest["selection_complete_without_errors"] = not manifest["errors"]
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
