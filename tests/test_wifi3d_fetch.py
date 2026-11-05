"""Ensure remote ZIP ranges cannot silently become multi-gigabyte downloads."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('fetch_wifi3d', Path(__file__).parents[1]/'scripts/fetch_wifi3d.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class Response:
    def __init__(self, status, headers, data=b''):
        self.status_code, self.headers, self.data = status, headers, data
        self.body_read = False

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass

    @property
    def content(self):
        self.body_read = True
        return self.data

    def iter_content(self, count):
        self.body_read = True
        yield self.data[:count]


class Client:
    def __init__(self, response): self.response, self.calls = response, []
    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_ignored_binary_range_is_refused_without_reading_archive_body():
    response = Response(200, {'Content-Type': 'application/zip'})
    client = Client(response)
    with pytest.raises(ValueError, match='Refusing a full download'):
        fetch.range_bytes(client, 10, 4)
    assert not response.body_read and len(client.calls) == 1


def test_custom_archive_requires_exact_range_identity_and_length():
    response = Response(206, {'Content-Range': 'bytes 10-13/100'}, b'test')
    client = Client(response)
    assert fetch.range_bytes(client, 10, 4, 'public-test-id', 100) == b'test'
    assert client.calls[0]['params']['id'] == 'public-test-id'
    assert client.calls[0]['headers']['Range'] == 'bytes=10-13'
    response.headers['Content-Range'] = 'bytes 10-13/101'
    with pytest.raises(ValueError):
        fetch.range_bytes(client, 10, 4, 'public-test-id', 100)
    with pytest.raises(ValueError, match='excessive'):
        fetch.range_bytes(client, 0, 150_000_001)


def test_quota_html_retries_are_bounded_and_never_parse_as_zip(monkeypatch):
    monkeypatch.setattr(fetch.time, 'sleep', lambda _: None)
    response = Response(200, {'Content-Type': 'text/html'}, b'<title>Google Drive - Quota exceeded</title>')
    client = Client(response)
    with pytest.raises(ValueError, match='Refusing a full download'):
        fetch.range_bytes(client, 10, 4)
    assert len(client.calls) == 3
