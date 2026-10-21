"""URL ingestion policy, provenance, cleanup and failures without live downloads."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reelkit import download, fsutil, store


class Download(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="reel-url-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        p = patch.object(store, "DATA", self.root / "data")
        p.start()
        self.addCleanup(p.stop)
        self.asset = Mock(id="asset", kind="source", meta={"sha256": "a" * 64})
        self.asset.cache.side_effect = lambda name: self.root / name

    def run_import(self, kind="source", result=None, error=None):
        options = {}
        def factory(opts):
            options.update(opts)
            def extract(*args, **kw):
                Path(opts["outtmpl"].replace("%(ext)s", "mkv")).write_bytes(b"downloaded")
                if error:
                    raise error
                return result if result is not None else {"id": "video", "title": "Talk", "height": 2160,
                                                          "format_id": "best-video+best-audio"}
            ydl = Mock()
            ydl.__enter__ = Mock(return_value=ydl)
            ydl.__exit__ = Mock(return_value=False)
            ydl.extract_info.side_effect = extract
            return ydl
        with patch.dict(sys.modules, {"yt_dlp": types.SimpleNamespace(YoutubeDL=factory)}), \
                patch.object(store, "ingest", return_value=self.asset) as ingest:
            asset = download.ingest_url("https://example.com/video", kind)
            ingest.assert_called_once()
        self.assertEqual(list((store.DATA / "imports").iterdir()), [])
        return options, asset

    def test_best_streams_are_preserved_and_provenance_recorded(self):
        options, asset = self.run_import()
        self.assertIs(asset, self.asset)
        self.assertEqual(options["format"], "bv*+ba/b")
        self.assertEqual(options["merge_output_format"], "mkv")
        self.assertNotIn("postprocessors", options)  # no lossy transcode
        self.assertEqual(fsutil.read_json(self.root / "import.json")["height"], 2160)

    def test_music_uses_best_audio(self):
        options, _ = self.run_import("music")
        self.assertEqual(options["format"], "ba/b")

    def test_download_failure_cleans_staging(self):
        with self.assertRaises(RuntimeError):
            self.run_import(error=RuntimeError("download interrupted"))
        self.assertEqual(list((store.DATA / "imports").iterdir()), [])
        self.assertFalse((self.root / "import.json").exists())

    def test_playlists_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_import(result={"_type": "playlist"})

    def test_non_http_input_is_rejected_before_downloading(self):
        for url in ("file:///tmp/video.mp4", "https://", "local.mp4"):
            with self.assertRaises(ValueError):
                download.ingest_url(url)


if __name__ == "__main__":
    unittest.main()
