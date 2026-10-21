"""Real child-process failures and atomic render output, without large fixtures."""
import sys
import tempfile
import time
import unittest
from pathlib import Path

from reelkit import ff, render


class ProcessFailures(unittest.TestCase):
    def test_failed_encoder_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "reel.mp4"
            out.write_bytes(b"previous accepted render")
            tl = {"width": 160, "height": 284, "fps": 30, "duration": 0.2,
                  "encode": {"preset": "not-a-real-preset"}}
            with self.assertRaises(render.RenderError) as cm:
                render.render(tl, out, timeout_s=10, stall_s=2)
            self.assertEqual(cm.exception.record["stage"], "video")
            self.assertEqual(out.read_bytes(), b"previous accepted render")
            self.assertEqual(list(Path(d).iterdir()), [out])

    def test_process_timeout_is_bounded_and_typed(self):
        start = time.monotonic()
        with self.assertRaises(ff.FFError) as cm:
            ff.run([sys.executable, "-c", "import time; time.sleep(60)"],
                   name="timeout fixture", timeout=0.15)
        self.assertEqual(cm.exception.reason, "timeout")
        self.assertTrue(cm.exception.record["transient"])
        self.assertLess(time.monotonic() - start, 5)

    def test_failed_child_reports_bounded_stderr(self):
        with self.assertRaises(ff.FFError) as cm:
            ff.run([sys.executable, "-c", "import sys; sys.stderr.write('x'*100000+'\\nactual failure\\n'); sys.exit(3)"],
                   name="exit fixture", timeout=5)
        self.assertEqual(cm.exception.record["returncode"], 3)
        self.assertIn("actual failure", cm.exception.record["stderr_tail"])
        self.assertLessEqual(len(cm.exception.record["stderr_tail"]), ff.TAIL_CHARS)


if __name__ == "__main__":
    unittest.main()
