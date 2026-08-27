"""Tests for LoaderTools.download's concurrency-safety fixes.

Regression guard for a real production incident: several dataset names
(RWTHLoader's acid_species/microgel_size families) share one upstream zip.
Concurrent cold-cache callers used to (a) all fire the request at once,
tripping the remote host's rate limiting (HTTP 429), and (b) race to write
the same destination file with no locking or atomicity, so a partial/error
response from one process could be read back as "done" by another --
surfacing downstream as CorruptedZipFileError.
"""

import glob
import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from raman_data.exceptions import CorruptedZipFileError
from raman_data.loaders.LoaderTools import LoaderTools


def _zip_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    return buf.getvalue()


def _fake_response(status_code=200, content=b"", headers=None):
    resp = MagicMock()
    resp.status_code = status_code

    def _raise_for_status():
        if status_code >= 400:
            raise requests.HTTPError(f"{status_code} error", response=resp)

    resp.headers = headers or {}
    resp.iter_content = lambda chunk_size: [content] if content else []
    resp.raise_for_status = _raise_for_status
    resp.close = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_download_writes_via_atomic_rename_no_leftover_tmp(tmp_path):
    content = _zip_bytes()
    resp = _fake_response(content=content, headers={"Content-Length": str(len(content))})

    with patch("raman_data.loaders.LoaderTools.requests.get", return_value=resp):
        out_path = LoaderTools.download(
            url="https://example.com/data.zip",
            out_dir_path=str(tmp_path),
            out_file_name="data.zip",
        )

    assert out_path == str(tmp_path / "data.zip")
    assert os.path.exists(out_path)
    assert LoaderTools.is_valid_zip(out_path)
    # No leftover .tmp<pid> files from the atomic-rename step.
    assert glob.glob(str(tmp_path / "*.tmp*")) == []


def test_download_skips_when_valid_file_already_present(tmp_path):
    dest = tmp_path / "data.zip"
    dest.write_bytes(_zip_bytes())

    with patch("raman_data.loaders.LoaderTools.requests.get") as mock_get:
        out_path = LoaderTools.download(
            url="https://example.com/data.zip",
            out_dir_path=str(tmp_path),
            out_file_name="data.zip",
        )

    mock_get.assert_not_called()
    assert out_path == str(dest)


def test_download_redownloads_when_existing_file_is_corrupt(tmp_path):
    dest = tmp_path / "data.zip"
    dest.write_bytes(b"<html>429 Too Many Requests</html>")  # not a real zip

    content = _zip_bytes()
    resp = _fake_response(content=content, headers={"Content-Length": str(len(content))})

    with patch("raman_data.loaders.LoaderTools.requests.get", return_value=resp) as mock_get:
        out_path = LoaderTools.download(
            url="https://example.com/data.zip",
            out_dir_path=str(tmp_path),
            out_file_name="data.zip",
        )

    mock_get.assert_called_once()
    assert LoaderTools.is_valid_zip(out_path)


def test_download_raises_and_cleans_up_tmp_on_corrupted_response(tmp_path):
    bad_content = b"<html>Not a zip</html>"
    resp = _fake_response(content=bad_content, headers={"Content-Length": str(len(bad_content))})

    with patch("raman_data.loaders.LoaderTools.requests.get", return_value=resp):
        with pytest.raises(CorruptedZipFileError):
            LoaderTools.download(
                url="https://example.com/data.zip",
                out_dir_path=str(tmp_path),
                out_file_name="data.zip",
            )

    assert not os.path.exists(tmp_path / "data.zip")
    assert glob.glob(str(tmp_path / "*.tmp*")) == []


def test_download_retries_on_429_then_succeeds(tmp_path):
    content = _zip_bytes()
    rate_limited = _fake_response(status_code=429, headers={"Retry-After": "0"})
    ok = _fake_response(content=content, headers={"Content-Length": str(len(content))})

    with patch("raman_data.loaders.LoaderTools.requests.get", side_effect=[rate_limited, ok]) as mock_get, \
            patch("raman_data.loaders.LoaderTools.time.sleep") as mock_sleep:
        out_path = LoaderTools.download(
            url="https://example.com/data.zip",
            out_dir_path=str(tmp_path),
            out_file_name="data.zip",
        )

    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()
    assert LoaderTools.is_valid_zip(out_path)


def test_download_gives_up_after_max_retries_on_persistent_429(tmp_path):
    rate_limited = _fake_response(status_code=429, headers={"Retry-After": "0"})

    with patch("raman_data.loaders.LoaderTools.requests.get", return_value=rate_limited) as mock_get, \
            patch("raman_data.loaders.LoaderTools.time.sleep"):
        with pytest.raises(requests.HTTPError):
            LoaderTools.download(
                url="https://example.com/data.zip",
                out_dir_path=str(tmp_path),
                out_file_name="data.zip",
            )

    assert mock_get.call_count == LoaderTools._MAX_RETRIES
