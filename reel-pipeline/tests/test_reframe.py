"""reframe.py: camera path behaviour (dead zone, no overshoot, smooth), identity tracking,
coverage clamping, and real face tracking on a moving face (if the test image exists)."""
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from reelkit import ff, reframe, render
from reelkit.paths import FFMPEG, MODELS

LENA = MODELS / "testdata" / "lena.jpg"


class Detection(unittest.TestCase):
    def test_confidence_comes_from_yunet_score_not_eye_coordinate(self):
        # Equal boxes: the more confident face must win regardless of eye x.
        faces = np.array([[10, 20, 40, 60, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8],
                          [10, 20, 40, 60, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]], np.float32)
        detector = Mock()
        detector.detect.return_value = (None, faces)
        with patch.object(reframe, "_detector", return_value=detector):
            got = reframe.detect_image(np.zeros((100, 200, 3), np.uint8))
        self.assertAlmostEqual(got[0][4], 0.8)
        self.assertAlmostEqual(got[1][4], 0.95)
        self.assertTrue(all(0 <= face[4] <= 1 for face in got))


class Path_(unittest.TestCase):
    def test_dead_zone_holds_then_follows_without_overshoot(self):
        # subject jitters ±0.02 around 0.5 for 2 s, then steps to 0.8
        track = [(t / 5, (0.5 + 0.02 * (-1) ** i, 0.5, 0.1, 0.1)) for i, t in enumerate(range(10))]
        track += [(2 + t / 5, (0.8, 0.5, 0.1, 0.1)) for t in range(15)]
        cam = reframe.path(track, fps=30, dead_zone=0.06, dur=5.0)
        xs = np.array([x for _, x, _ in cam])
        self.assertLess(np.ptp(xs[:55]), 1e-6, "jitter inside the dead zone must not move the camera")
        self.assertLessEqual(xs.max(), 0.8 - 0.06 + 1e-6, "no overshoot past the dead-zone goal")
        self.assertAlmostEqual(xs[-1], 0.74, delta=0.01)
        self.assertLess(np.abs(np.diff(xs, 2)).max(), 0.004, "acceleration stays smooth")

    def test_no_subject_gives_none(self):
        self.assertIsNone(reframe.path([(0, None), (0.2, None)], dur=1.0))

    def test_identity_is_kept_through_a_bigger_passerby(self):
        a, b = (0.3, 0.5, 0.10, 0.12, 0.9), (0.75, 0.5, 0.14, 0.16, 0.9)
        dets = [(i * 0.2, [a]) for i in range(5)] + [(1.0 + i * 0.2, [a, b]) for i in range(3)]
        tr = reframe.subject(dets, switch_after=1.5, dominance=1.8)
        self.assertTrue(all(abs(p[0] - 0.3) < 1e-6 for _, p in tr), "brief, not dominant: stay on A")
        big = (0.75, 0.5, 0.3, 0.32, 0.95)
        dets2 = [(i * 0.2, [a]) for i in range(5)] + [(1.0 + i * 0.2, [a, big]) for i in range(12)]
        tr2 = reframe.subject(dets2, switch_after=1.5, dominance=1.8)
        self.assertAlmostEqual(tr2[-1][1][0], 0.75, delta=1e-6, msg="a clearly dominant face for > 1.5 s takes over")
        self.assertAlmostEqual(tr2[6][1][0], 0.3, delta=1e-6)

    def test_offsets_are_clamped_to_keep_the_canvas_covered(self):
        cam = [(0.0, 0.0, 0.5), (1.0, 1.0, 0.5)]           # subject at the far left / right edge
        xk, _ = reframe.offsets(cam, 1920, 1080, (720, 1280), lambda t: 1.0)
        limit = (1920 * 1280 / 1080 - 720) / 2
        self.assertAlmostEqual(xk[0][1], limit, delta=0.01)
        self.assertAlmostEqual(xk[-1][1], -limit, delta=0.01)
        xk2, _ = reframe.offsets([(0.0, 0.5, 0.5), (1.0, 0.6, 0.5)], 1920, 1080, (720, 1280), lambda t: 1.0)
        self.assertEqual(len(xk2), 2, "a linear move needs only its end keys")


@unittest.skipUnless(LENA.exists(), "face test image not downloaded")
class Faces(unittest.TestCase):
    def test_tracks_a_moving_face_into_the_centre_of_a_vertical_crop(self):
        tmp = Path(tempfile.mkdtemp(prefix="rftest-"))
        vid = tmp / "face.mp4"
        # 1920×1080: the face (512 px image) slides 250 px/s for 2.5 s, then stops
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x404850:s=1920x1080:r=30:d=4",
                        "-i", str(LENA), "-filter_complex", "[0][1]overlay=x='200+250*min(t\\,2.5)':y=284",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(vid)], check=True)
        rk = reframe.clip_keyframes(str(vid), 0.0, 4.0, (720, 1280), lambda t: 1.0)
        self.assertTrue(rk["subject"])
        self.assertEqual(rk["y"], 0)
        f = 1280 / 1080
        for t in (0.0, 1.0, 2.0, 2.5, 3.9):
            x = render.kf(rk["x"], t)
            face_x = (200 + 250 * min(t, 2.5) + 256) / 1920   # where the face is in the source (0..1)
            on_canvas = 360 + x + (face_x - 0.5) * 1920 * f
            # moving: the whole face (≈ 175 px on canvas) stays inside the 720 px crop
            self.assertLess(abs(on_canvas - 360), 360 - 90, f"t={t}: face at {on_canvas:.0f}px")
            if t > 3.5:   # 1.4 s after it stops: settled within the dead zone
                self.assertLess(abs(on_canvas - 360), 0.06 * 1920 * f + 15, f"t={t}: face at {on_canvas:.0f}px")


if __name__ == "__main__":
    unittest.main()

