"""Optional yt-dlp ingestion. Preserve the best available streams, then hand
the completed local file to the same verified asset store as uploaded footage."""
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from . import ff, fsutil, store
from .paths import FFMPEG


class _Logger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        ff.log(f"yt-dlp: {msg}")

    error = warning


def ingest_url(url, kind="source", max_bytes=None):
    """Download one video/audio URL without transcoding, ingest and clean staging.
    Returns an Asset. yt-dlp remains optional for callers using local files."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("a complete http(s) media URL is required")
    if kind not in ("sample", "source", "music", "sfx"):
        raise ValueError("URL imports support sample, source, music or sfx")
    if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer")
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("URL imports need yt-dlp: pip install -r requirements-url.txt") from None
    staging = store.DATA / "imports"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=staging) as tmp:
        options = {"format": "ba/b" if kind in ("music", "sfx") else "bv*+ba/b",
                   "outtmpl": str(Path(tmp) / "input.%(ext)s"), "merge_output_format": "mkv",
                   "ffmpeg_location": str(Path(FFMPEG).parent),
                   "noplaylist": True, "playlist_items": "1", "quiet": True, "noprogress": True,
                   "socket_timeout": 30, "retries": 3, "fragment_retries": 3, "logger": _Logger()}
        if max_bytes is not None:
            received = {}
            def budget(event):
                received[event.get("filename", "download")] = event.get("downloaded_bytes") or 0
                if sum(received.values()) > max_bytes:
                    raise RuntimeError(f"download exceeds the {max_bytes} byte budget")
            options.update(max_filesize=max_bytes, progress_hooks=[budget])
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise RuntimeError(f"media download failed: {e}") from e
        if not isinstance(info, dict) or info.get("_type", "video") != "video":
            raise ValueError("import one video URL; playlists and collections are not supported")
        files = [p for p in Path(tmp).glob("input.*") if p.is_file() and p.suffix.lower() in
                 (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac")]
        if len(files) != 1:
            raise RuntimeError("yt-dlp did not produce exactly one completed media file")
        if max_bytes is not None and files[0].stat().st_size > max_bytes:
            raise RuntimeError(f"download exceeds the {max_bytes} byte budget")
        asset = store.ingest(files[0], kind)
        fsutil.atomic_write_json(asset.cache("import.json"), {
            "url": info.get("webpage_url") or url, "title": info.get("title"),
            "extractor": info.get("extractor"), "media_id": info.get("id"),
            "format_id": info.get("format_id"), "width": info.get("width"), "height": info.get("height"),
            "fps": info.get("fps"), "vcodec": info.get("vcodec"), "acodec": info.get("acodec"),
            "sha256": asset.meta["sha256"]})
        return asset
