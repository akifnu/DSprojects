"""Prove that preview creation is cheap and approval cannot leak across edits."""
import copy
import json
import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image, ImageChops, ImageSequence

from reelkit import asr, batch, captions, fsutil, orchestrator as O, source, store, storyboard
from reelkit.paths import FFMPEG


class Storyboards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = tempfile.TemporaryDirectory(prefix="storyboard-source-")
        cls.src = Path(cls.media.name) / "source.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x568:r=30:d=5",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=5",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(cls.src)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.media.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="storyboard-job-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for mod in (O, store, batch):
            p = patch.object(mod, "DATA", self.root / "data")
            p.start()
            self.addCleanup(p.stop)
        self.asset = store.ingest(self.src, "source")
        words = [{"w": w, "s": 0.1 + i * 0.5, "e": 0.5 + i * 0.5}
                 for i, w in enumerate("First complete thought. Second complete payoff.".split())]
        tr = {"lang": "en", "timing": "word", "fingerprint": asr.fingerprint("en"), "duration": 5,
              "words": words, "sentences": asr.sentences(words)}
        fsutil.atomic_write_json(self.asset.cache("transcript.en.json"), tr)
        style = {"version": 2, "sample": {"id": self.asset.id, "width": 320, "height": 568, "duration": 5},
                 "shots": [], "pacing": {"max_pause": 0.3}, "motion": {}, "transitions": {},
                 "captions": {"present": True, "style": {"font": "Arimo", "size": 28, "y": 40,
                                                           "box": {"color": "#000000"}}}}
        self.jid = O.create(name="preview", execution={"encoder": "cpu"})["id"]
        with O._txn(self.jid) as job:
            job["inputs"]["samples"] = [self.asset.id]
            job["inputs"]["sources"] = [self.asset.id]
            job["steps"]["intake"] = {"done": True}
            O._publish_style(job, style, self.asset.id, "fixture", "fixture")
            for step in ("analyze", "transcribe"):
                job["steps"][step] = {"done": True, "key": O._key(job, step)}
            plan = {"name": self.jid, "style": job["style_version"]["path"], "sources": {"S": self.asset.id},
                    "output": {"width": 360, "height": 640},
                    "reels": [{"id": "r1", "title": "First", "segments": [{"src": "S", "sentences": [0, 0]}]},
                              {"id": "r2", "title": "Second", "segments": [{"src": "S", "sentences": [1, 1]}]}]}
            errors, prep = O._check_plan(job, plan, None)
            self.assertFalse(errors, errors)
            O._accept_plan(job, prep, "fixture-plan", "fixture")

    def preview(self):
        with patch.object(source, "prepare", side_effect=AssertionError("editing copy before approval")), \
                patch.object(source, "audio", side_effect=AssertionError("audio before approval")), \
                patch.object(source, "look", side_effect=AssertionError("full colour analysis before approval")), \
                patch("reelkit.render.render", side_effect=AssertionError("video encode before approval")), \
                patch.object(asr, "transcribe", side_effect=AssertionError("repeated ASR")):
            job = O.advance(self.jid)
        self.assertEqual(job["status"], "needs:approve_storyboard", job.get("error"))
        return job

    def decide(self, job, decision="approve", notes=""):
        n = job["needs"]
        return O.decide_storyboard(self.jid, decision, n["need_id"], n["revision"], n["storyboard_sha256"], notes)

    def test_preview_blocks_all_expensive_work_and_reuses_itself(self):
        job = self.preview()
        self.assertEqual(job["storyboard"]["frames"], 6)
        self.assertLess(job["storyboard"]["frames"], job["storyboard"]["estimated_render_frames"])
        self.assertTrue(Path(job["needs"]["page"]).is_file())
        self.assertTrue(storyboard.intact(job["storyboard"]))
        self.assertEqual(len(job["needs"]["gifs"]), 2)
        self.assertGreater(job["storyboard"]["gif_frames"], 2)
        for gif in job["storyboard"]["gifs"]:
            self.assertLessEqual(gif["frames"], 18)
            self.assertLessEqual(gif["width"], 240)
            self.assertLessEqual(gif["bytes"], 2 * 1024 * 1024)
            with Image.open(gif["file"]) as im:
                self.assertEqual(im.format, "GIF")
                self.assertEqual(im.n_frames, gif["frames"])
                self.assertEqual(im.info["loop"], 0)
                frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
                # Ignore the timestamp ribbon: the footage itself must move.
                region = (0, 18, im.width, im.height)
                self.assertIsNotNone(ImageChops.difference(frames[0].crop(region), frames[-1].crop(region)).getbbox())
                im.seek(0)
                durations = [f.info["duration"] for f in ImageSequence.Iterator(im)]
                self.assertAlmostEqual(sum(durations) / 1000, gif["duration"])
        page = Path(job["needs"]["page"]).read_text()
        self.assertIn("Download GIF", page)
        self.assertIn("Source range", page)
        self.assertEqual(self.preview()["needs"]["need_id"], job["needs"]["need_id"])
        for step in ("prepare", "render"):
            with self.assertRaises(O.JobError) as cm:
                O._context(job, step)
            self.assertEqual(cm.exception.code, "approval_required")

    def test_only_explicit_current_approval_unlocks_full_work(self):
        job = self.preview()
        n = job["needs"]
        for nid, rev, sha in (("old", n["revision"], n["storyboard_sha256"]),
                              (n["need_id"], -1, n["storyboard_sha256"]),
                              (n["need_id"], n["revision"], "wrong")):
            with self.assertRaises(O.Rejected):
                O.decide_storyboard(self.jid, "approve", nid, rev, sha)
        with self.assertRaises(O.Rejected):
            O.complete(self.jid, "approve_storyboard", need_id=n["need_id"], ok=["all"])
        self.decide(job)
        accepted = O._load(self.jid)
        self.assertTrue(O._valid(accepted, "approve_storyboard"))
        self.assertEqual(O._next_step(accepted), "prepare")
        self.assertEqual(O._context(accepted, "render")["storyboard"]["sha256"], n["storyboard_sha256"])

    def test_feedback_reopens_plan_and_only_changed_reel_rebuilds(self):
        job = self.preview()
        self.decide(job, "revise", "Change the first hook")
        revised = O.advance(self.jid)
        self.assertEqual(revised["needs"]["step"], "plan")
        self.assertIn("Change the first hook", revised["needs"]["storyboard_feedback"][-1]["notes"])
        plan = fsutil.read_json(revised["needs"]["plan"])
        plan["reels"][0]["title_text"] = "A new hook"
        fsutil.atomic_write_json(revised["needs"]["plan"], plan)
        need = revised["needs"]
        O.complete(self.jid, "plan", need_id=need["need_id"], revision=need["revision"])
        fresh = self.preview()
        self.assertEqual(fresh["storyboard"]["cache_hits"], 1)
        self.assertNotEqual(job["storyboard"]["sha256"], fresh["storyboard"]["sha256"])
        self.assertFalse(O._valid(fresh, "approve_storyboard"))
        self.assertEqual(fresh["review"]["rounds"], 0)
        with self.assertRaises(O.Rejected):
            self.decide(job)

    def test_changed_preview_or_transcript_invalidates_approval(self):
        job = self.preview()
        manifest = fsutil.read_json(job["storyboard"]["manifest"])
        Path(manifest["reels"][0]["frames"][0]["file"]).write_bytes(b"tampered")
        with self.assertRaises(O.Rejected):
            self.decide(job)
        fresh = self.preview()
        self.assertEqual(fresh["storyboard"]["cache_hits"], 2)
        self.decide(fresh)
        with O._txn(self.jid) as current:
            for step in ("prepare", "render"):
                current["steps"][step] = {"done": True, "key": O._key(current, step)}
        tr_path = self.asset.cache("transcript.en.json")
        tr = fsutil.read_json(tr_path)
        tr["words"][0]["w"] = "Changed"
        fsutil.atomic_write_json(tr_path, tr)
        with self.assertRaises(O.JobError):
            O._context(O._load(self.jid), "render")
        updated = self.preview()
        self.decide(updated)
        current = O._load(self.jid)
        self.assertEqual(O._next_step(current), "prepare")
        self.assertFalse(O._valid(current, "render"))

    def test_budget_limit_fails_before_a_single_frame_is_generated(self):
        with O._txn(self.jid) as job:
            job["policy"]["storyboard"]["max_frames"] = 1
        with patch.object(storyboard, "_frame", side_effect=AssertionError("budget ignored")) as frame:
            job = O.advance(self.jid)
        frame.assert_not_called()
        self.assertEqual(job["status"], "failed")
        self.assertIn("exceeds", job["error"]["cause"])
        self.assertNotIn("render", job["steps"])

    def test_gif_budget_fails_before_composing_any_preview(self):
        with O._txn(self.jid) as job:
            job["policy"]["storyboard"]["max_gif_frames"] = 1
        with patch.object(storyboard, "_frame") as still, patch.object(storyboard, "_gif") as gif:
            job = O.advance(self.jid)
        still.assert_not_called()
        gif.assert_not_called()
        self.assertEqual(job["status"], "failed")
        self.assertIn("GIF frames", job["error"]["cause"])
        self.assertIsNone(job["storyboard_approval"])

    def test_time_budget_stops_before_composing(self):
        job = O._load(self.jid)
        ticks = iter([0, 2])
        with patch.object(storyboard, "time", SimpleNamespace(monotonic=lambda: next(ticks))), \
                patch.object(storyboard, "_frame") as frame:
            with self.assertRaisesRegex(storyboard.StoryboardError, "time budget"):
                storyboard.build(job["plan"]["path"], self.root / "budget", {"timeout_s": 1})
        frame.assert_not_called()

    def test_gif_byte_budget_cannot_publish_an_approvable_board(self):
        with O._txn(self.jid) as job:
            job["policy"]["storyboard"]["max_gif_bytes"] = 1024
        job = O.advance(self.jid)
        self.assertEqual(job["status"], "failed")
        self.assertIn("GIF exceeds", job["error"]["cause"])
        self.assertIsNone(job["storyboard"])

    def test_gif_tampering_blocks_approval_and_reuses_the_intact_cache(self):
        job = self.preview()
        Path(job["needs"]["gifs"][0]["file"]).write_bytes(b"tampered GIF")
        with self.assertRaises(O.Rejected):
            self.decide(job)
        fresh = self.preview()
        self.assertEqual(fresh["storyboard"]["cache_hits"], 2)
        self.assertTrue(storyboard.intact(fresh["storyboard"]))
        with Image.open(fresh["needs"]["gifs"][0]["file"]) as gif:
            self.assertGreater(gif.n_frames, 1)

    def test_gif_cache_damage_rebuilds_only_the_damaged_reel(self):
        job = self.preview()
        cache = O._dir(self.jid) / "storyboards" / "cache"
        next(cache.glob("*/preview.gif")).write_bytes(b"broken cache")
        # Force a fresh board, leaving the independent cache of the other reel intact.
        Path(job["needs"]["sheet"]).unlink()
        fresh = self.preview()
        self.assertEqual(fresh["storyboard"]["cache_hits"], 1)
        self.assertTrue(storyboard.intact(fresh["storyboard"]))

    def test_new_plan_invalidates_an_approved_storyboard(self):
        job = self.preview()
        self.decide(job)
        with O._txn(self.jid) as current:
            plan = fsutil.read_json(current["plan"]["path"])
            plan["reels"][0]["emphasis"] = ["thought"]
            errors, prep = O._check_plan(current, plan, None)
            self.assertFalse(errors)
            O._accept_plan(current, prep, "revision", "changed emphasis")
        self.assertFalse(O._valid(O._load(self.jid), "approve_storyboard"))
        self.assertEqual(self.preview()["storyboard"]["cache_hits"], 1)

    def test_work_context_rechecks_artifacts_before_expensive_work(self):
        job = self.preview()
        self.decide(job)
        ctx = O._context(O._load(self.jid), "prepare")
        Path(ctx["storyboard"]["sheet"]).write_bytes(b"changed after approval")
        with self.assertRaises(O.JobError) as cm:
            O._w_prepare(ctx)
        self.assertEqual(cm.exception.code, "approval_required")

    def test_replacing_a_font_invalidates_approval(self):
        font = self.root / "preview-font.ttf"
        shutil.copyfile(captions.font_file("Arimo", 800), font)
        with patch.object(captions, "font_file", return_value=str(font)):
            job = self.preview()
            self.decide(job)
            with font.open("ab") as out:
                out.write(b"replacement-font-revision")
            self.assertFalse(O._valid(O._load(self.jid), "approve_storyboard"))
            with self.assertRaises(O.JobError):
                O._context(O._load(self.jid), "render")

    def test_existing_jobs_gain_a_gate_without_automatic_approval(self):
        job = O._load(self.jid)
        for k in ("storyboard", "storyboard_approval", "storyboard_feedback"):
            job.pop(k, None)
        job["policy"].pop("storyboard", None)
        job["policy"]["version"] = 2
        upgraded = O._migrate(job)
        self.assertEqual(upgraded["policy"]["version"], 4)
        self.assertIsNone(upgraded["storyboard_approval"])
        self.assertEqual(O._next_step(upgraded), "storyboard")

    def test_existing_storyboard_policy_keeps_its_limits_and_adds_gifs(self):
        old = copy.deepcopy(O.POLICY)
        old["version"] = 3
        old["storyboard"] = {"width": 180, "frames_per_reel": 2, "max_frames": 90, "timeout_s": 30}
        migrated = O._migrate_policy(old)
        self.assertEqual(migrated["storyboard"]["width"], 180)
        self.assertEqual(migrated["storyboard"]["timeout_s"], 30)
        self.assertEqual(migrated["storyboard"]["gif_fps"], 6)
        self.assertIn("storyboard.gif_fps", migrated["filled_defaults"])


class Workflow(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js not installed (saved JS workflow check)")
    def test_workflow_returns_to_user_instead_of_auto_approving(self):
        script = Path(__file__).with_name("workflow_storyboard.mjs")
        p = subprocess.run(["node", str(script)], text=True, capture_output=True, timeout=10)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
