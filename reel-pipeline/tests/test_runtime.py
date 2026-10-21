"""Job lifecycle with stubbed analysis/ASR/judgments and real ingest, render, QC
and delivery; state recovery, review fencing, cache identities and encoder policy."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reelkit import asr, batch, fsutil, orchestrator as O, readiness, source, store
from reelkit.paths import FFMPEG


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="reel-runtime-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module in (O, batch, store):
            p = patch.object(module, "DATA", self.root / "data")
            p.start()
            self.addCleanup(p.stop)

    def job(self):
        return O.create(name="runtime", execution={"encoder": "cpu"})

    def test_full_job_with_stubbed_analysis_and_agent_answers(self):
        src = self.root / "source.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x284:r=30:d=3",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=3",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(src)], check=True)
        asset = store.ingest(src, "source")
        words = [{"w": w, "s": 0.2 + i * 0.4, "e": 0.5 + i * 0.4}
                 for i, w in enumerate("One complete idea with a payoff.".split())]
        tr = {"lang": "en", "timing": "word", "fingerprint": asr.fingerprint("en"),
              "duration": 3, "words": words, "sentences": asr.sentences(words)}
        fsutil.atomic_write_json(asset.cache("transcript.en.json"), tr)
        style = {"version": 2, "sample": {"id": asset.id, "width": 160, "height": 284, "duration": 3, "fps": 30},
                 "shots": [], "pacing": {"median_shot": 1.5, "max_pause": 0.25}, "motion": {}, "transitions": {},
                 "captions": {"present": False}, "audio": {"loudness": {"I": -14}}, "grade": {}}

        def analyze(ctx):
            fsutil.atomic_write_json(asset.cache("style.json"), style)
            return {asset.id: []}

        job = O.create(name="runtime", roles={"source": [src], "sample": [src]}, count=1,
                       execution={"encoder": "cpu", "jobs": 1})
        jid = job["id"]
        with O._txn(jid) as job:
            job["policy"].update(jobs=1, render_quality="draft")
        with patch.dict(O._WORK, {"analyze": analyze}):
            job = O.advance(jid)
            self.assertEqual(job["needs"]["step"], "verify_style")
            need = job["needs"]
            verified = copy.deepcopy(style)
            verified["review"] = {"verified": True}
            fsutil.atomic_write_json(need["style"], verified)
            O.complete(jid, "verify_style", need_id=need["need_id"], revision=need["revision"])

            job = O.advance(jid)
            self.assertEqual(job["needs"]["step"], "plan")
            need = job["needs"]
            plan = fsutil.read_json(need["plan"])
            plan["output"].update(width=160, height=284, quality="draft")
            key = next(iter(plan["sources"]))
            plan["reels"] = [{"id": "r1", "segments": [{"src": key, "sentences": [0, 0]}]}]
            fsutil.atomic_write_json(need["plan"], plan)
            O.complete(jid, "plan", need_id=need["need_id"], revision=need["revision"])

            # The storyboard may use the compositor, but never full media work.
            with patch.object(source, "prepare", side_effect=AssertionError("prepared before approval")), \
                    patch("reelkit.render.render", side_effect=AssertionError("rendered before approval")):
                job = O.advance(jid)
                self.assertEqual(job["needs"]["step"], "approve_storyboard", job.get("error"))
                self.assertTrue(Path(job["needs"]["sheet"]).is_file())
                current = job["needs"]
                self.assertEqual(O.advance(jid)["needs"]["need_id"], current["need_id"])
            O.decide_storyboard(jid, "approve", current["need_id"], current["revision"], current["storyboard_sha256"])
            job = O.advance(jid)
            self.assertEqual(job["needs"]["step"], "review", job.get("error"))
            need = job["needs"]
            answer = {"need_id": need["need_id"], "results": [
                {"reel_id": a["id"], "render_sha256": a["render_sha256"], "decision": "accept", "findings": []}
                for a in need["assigned"]]}
            fsutil.atomic_write_json(need["result_file"], answer)
            O.complete(jid, "review", need_id=need["need_id"], revision=need["revision"], result_file=need["result_file"])
            # Repeating exactly the accepted answer is safe.
            self.assertTrue(O.complete(jid, "review", need_id=need["need_id"],
                                       result_file=need["result_file"])["idempotent"])
            job = O.advance(jid)
        self.assertEqual(job["status"], "succeeded", job.get("deliver"))
        self.assertEqual(job["deliver"]["counts"]["delivered"], 1)
        self.assertTrue(Path(job["deliver"]["files"][0]).is_file())
        self.assertEqual(O.wait(jid, timeout=0)[0], 0)

    def test_stale_answers_cannot_advance_the_job(self):
        jid = self.job()["id"]
        with O._txn(jid) as job:
            O._issue(job, "plan")
        job = O._load(jid)
        with self.assertRaises(O.Rejected):
            O.complete(jid, "plan", need_id="old-need")
        with self.assertRaises(O.Rejected):
            O.complete(jid, "plan", need_id=job["needs"]["need_id"], revision=-1)
        self.assertEqual(O._load(jid)["needs"], job["needs"])

    def test_corrupt_current_state_recovers_previous_revision(self):
        jid = self.job()["id"]
        with O._txn(jid) as job:
            job["status"] = "waiting:worker"
        O._path(jid).write_text("{broken")
        self.assertEqual(O._load(jid)["id"], jid)
        O._path(jid).with_name("job.prev.json").write_text("{broken")
        with self.assertRaises(O.JobError) as cm:
            O._load(jid)
        self.assertEqual(cm.exception.code, "corrupt_state")

    def test_dead_worker_is_reported_and_can_be_restarted(self):
        jid = self.job()["id"]
        _, att, rec = O._claim(jid, foreground=True)
        rec.update(pid=99999999, acknowledged=True, stage="render")
        fsutil.atomic_write_json(O._att_path(jid, att), rec)
        code, status = O.wait(jid, timeout=0)
        self.assertEqual(code, 2)
        self.assertEqual(status["status"], "waiting:worker")
        self.assertEqual(status["error"]["cause"], "worker_died")
        kind, replacement, _ = O._claim(jid, foreground=True)
        self.assertEqual(kind, "new")
        self.assertNotEqual(replacement, att)

    def test_cancel_retry_and_attempt_fencing(self):
        jid = self.job()["id"]
        self.assertEqual(O.cancel(jid)["status"], "cancelled")
        self.assertFalse(O.retry(jid)["terminal"])
        self.assertFalse(O._cancel_flag(jid).exists())
        with O._txn(jid) as job:
            job["attempt"] = "replacement"
        with self.assertRaises(O._Superseded):
            with O._txn(jid, attempt="old"):
                self.fail("stale worker acquired the job")

    def test_transient_stage_retries_are_bounded(self):
        jid = self.job()["id"]
        err = {"stage": "render", "cause": "timeout", "transient": True}
        with O._txn(jid) as job:
            O._step_failed(job, "render", err)
            self.assertFalse(job["terminal"])
            O._step_failed(job, "render", err)
            self.assertTrue(job["terminal"])
            self.assertEqual(job["status"], "failed")

    def test_review_acceptance_cannot_waive_failed_technical_qc(self):
        job = O._blank("runtime")
        need = {"need_id": "review-1", "assigned": [{"id": "r1", "render_sha256": "a" * 64,
                                                       "qc_verdict": "fail"}]}
        errors, _ = O._check_review(job, need, {"results": [{"reel_id": "r1", "decision": "accept"}]})
        self.assertTrue(any("cannot be waived" in e for e in errors), errors)

    def test_stale_render_review_is_rejected(self):
        job = O._blank("runtime")
        need = {"need_id": "review-1", "assigned": [{"id": "r1", "render_sha256": "a" * 64,
                                                       "qc_verdict": "pass"}]}
        errors, _ = O._check_review(job, need, {"results": [{"reel_id": "r1", "render_sha256": "b" * 64,
                                                             "decision": "accept"}]})
        self.assertTrue(any("stale review" in e for e in errors), errors)

    def test_tampered_deliverable_cannot_be_delivered(self):
        jid = self.job()["id"]
        out = self.root / "reel.mp4"
        out.write_bytes(b"reviewed render")
        digest = fsutil.sha256_file(out)
        manifest = {"generation": 1, "reels": [{"id": "r1", "status": "accepted", "qc_verdict": "pass",
                                                "render_sha256": digest}],
                    "deliverables": [{"id": "r1", "file": str(out), "sha256": digest}]}
        mp = self.root / "manifest.json"
        fsutil.atomic_write_json(mp, manifest)
        out.write_bytes(b"changed after review")
        with O._txn(jid) as job:
            job["policy"]["spot_check"] = "none"
            job["plan"] = {"reels": ["r1"]}
            job["render"] = {"manifest": str(mp)}
            ctx = O._context(job, "deliver")
            O._c_deliver(job, ctx, O._w_deliver(ctx))
            self.assertEqual(job["status"], "needs_attention")
            self.assertFalse(job["deliver"]["files"])


class EncoderPolicy(unittest.TestCase):
    def test_unavailable_gpu_falls_back_only_when_allowed(self):
        with patch.object(readiness, "_nvenc", return_value=(False, "driver missing", None)):
            self.assertEqual(readiness.resolve_encoder("auto")["actual"], "cpu")
            for kw in ({"require_gpu": True}, {"allow_cpu_fallback": False}):
                with self.assertRaises(readiness.NotReady):
                    readiness.resolve_encoder("auto", **kw)

    def test_probe_exception_does_not_claim_gpu_readiness(self):
        with patch.object(readiness, "_nvenc", side_effect=RuntimeError("probe broke")):
            with self.assertRaises(readiness.NotReady):
                readiness.resolve_encoder("nvenc", require_gpu=True)


class CacheIdentity(unittest.TestCase):
    def test_corrupt_render_and_qc_are_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out = d / "r1.mp4"
            out.write_bytes(b"original")
            rec = {"render_key": "key", "plan_entry_hash": "entry", "render_sha256": fsutil.sha256_file(out)}
            self.assertIsNotNone(batch._render_current(rec, "key", "entry", out)[0])
            self.assertIsNone(batch._render_current(rec, "new-key", "entry", out)[0])
            out.write_bytes(b"tampered")
            self.assertTrue(batch._render_current(rec, "key", "entry", out)[2])
            q = d / "qc.json"
            fsutil.atomic_write_json(q, {"verdict": "pass"})
            rec.update(qc_key="qc", qc_sha256=fsutil.sha256_file(q))
            self.assertIsNotNone(batch._qc_current(rec, "qc", d)[0])
            self.assertIsNone(batch._qc_current(rec, "changed-policy", d)[0])
            q.write_text("{broken")
            self.assertIsNone(batch._qc_current(rec, "qc", d)[0])


if __name__ == "__main__":
    unittest.main()
