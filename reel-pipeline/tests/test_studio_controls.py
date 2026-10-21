"""Per-reel UI controls must reach the same compiler used for final exports."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import test_storyboard as fixtures
from reelkit import asr, batch, compile as C, contracts, fsutil, plan as P, render, source


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.Storyboards.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.Storyboards.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.Storyboards()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        job = fixtures.O._load(self.fixture.jid)
        self.plan = fsutil.read_json(job["plan"]["path"])
        self.style = fsutil.read_json(self.plan["style"])
        self.tr = fsutil.read_json(self.fixture.asset.cache("transcript.en.json"))

    def compile(self, reel, preview=False):
        a = self.fixture.asset
        path = C.compile_reel(self.plan, reel, self.style, self.fixture.root / "compiled",
                 transcripts={"S": self.tr}, envelopes={}, reframing=False, preview=preview,
                 copies={"S": [{"s": 0, "e": 5, "path": str(a.path), "width": a.info["width"],
                                "height": a.info["height"], "fps": a.info["fps"]}]})
        return fsutil.read_json(path)

    def test_caption_and_crop_controls_reach_the_final_timeline(self):
        reel = copy.deepcopy(self.plan["reels"][0])
        reel["editing"] = {"captions": {"style": {"fill": "#ffe600", "size": 64}},
                           "crop": {"x": .7, "y": .4, "zoom": 1.4}}
        tl = self.compile(reel)
        self.assertEqual(tl["styles"]["caption"]["fill"], "#ffe600")
        self.assertAlmostEqual(tl["styles"]["caption"]["size"], 64 * 640 / 568)
        self.assertTrue(all(c["meta"]["manual_crop"] for c in tl["video"]))
        self.assertGreaterEqual(tl["video"][0]["scale"], 1.4)
        self.assertNotEqual(tl["video"][0]["x"], [[0, 0]])
        self.assertTrue(any(c["role"] == "voice" for c in tl["audio"]["clips"]))
        out = self.fixture.root / "controlled.jpg"
        render.still(tl, .5, out)
        self.assertGreater(out.stat().st_size, 1000)
        self.assertEqual(self.style["captions"]["style"]["size"], 28)

    def test_disabling_captions_removes_final_caption_events(self):
        reel = dict(self.plan["reels"][0], editing={"captions": {"enabled": False}})
        self.assertEqual(self.compile(reel)["captions"], [])

    def test_cached_transcript_validation_does_not_prepare_source_audio(self):
        with patch.object(source, "audio", side_effect=AssertionError("audio preparation for cached transcript")), \
                patch.object(asr, "transcribe", side_effect=AssertionError("repeat transcription")):
            tr = source.transcript(self.fixture.asset, "en", "cpu")
        self.assertEqual(tr["words"], self.tr["words"])

    def test_nested_style_edits_preserve_unmentioned_reference_values(self):
        self.style["captions"]["style"]["anim"] = {"in": "pop", "in_dur": .2, "overshoot": .1}
        result = C.reel_style(self.style, {"editing": {"captions": {"style": {"anim": {"in": "fade"}}}}})
        self.assertEqual(result["captions"]["style"]["anim"], {"in": "fade", "in_dur": .2, "overshoot": .1})
        self.assertEqual(self.style["captions"]["style"]["anim"]["in"], "pop")

    def test_source_qualified_caption_edits_do_not_leak_into_another_source(self):
        reel = {"id": "mixed", "segments": [{"src": "A", "sentences": [0]}, {"src": "B", "sentences": [0]}],
                "caption_text": {"A:0": "The edited words."}}
        self.assertEqual(contracts.validate_plan({**self.plan, "sources": {"A": self.fixture.asset.id,"B":self.fixture.asset.id},
                                                "reels": [reel]})["errors"], [])
        runs = P.resolve_reel(reel, {"A": self.tr, "B": self.tr})
        a = " ".join(w["w"] for r in runs if r["src"] == "A" for w in r["words"])
        b = " ".join(w["w"] for r in runs if r["src"] == "B" for w in r["words"])
        self.assertEqual(a, "The edited words.")
        self.assertEqual(b, "First complete thought.")

    def test_sfx_gain_override_takes_precedence_over_measured_reference_levels(self):
        reel = dict(self.plan["reels"][0], editing={"sfx": {"gain_db": -55}})
        profile = {"caption": {"rate": 1, "kind": "pop", "offset": 0, "level_db": -8}}
        with patch.object(C, "_sfx_profile", return_value=profile), \
                patch.object(C, "_sfx_path", return_value=str(self.fixture.asset.path)), \
                patch.object(C, "_sound", return_value=(.15, 0, -30)), patch.object(C, "_voice_db", return_value=-20):
            tl = self.compile(reel)
        effects = [a for a in tl["audio"]["clips"] if a["role"] == "sfx"]
        self.assertTrue(effects)
        self.assertEqual({a["gain_db"] for a in effects}, {-55})

    def test_invalid_and_unknown_controls_are_rejected_before_rendering(self):
        for editing in ({"crop": {"zoom": 4}}, {"captions": {"style": {"size": 100000}}},
                        {"unknown_effect": True}, {"music": {"gain_db": 100}}):
            plan = copy.deepcopy(self.plan)
            plan["reels"][0]["editing"] = editing
            self.assertTrue(contracts.validate_plan(plan)["errors"])

    def test_final_batch_uses_effective_qc_style_and_reuses_edited_render_cache(self):
        self.plan["reels"] = [self.plan["reels"][0]]
        self.plan["reels"][0]["editing"] = {"captions": {"style": {"fill": "#ffe600", "size": 40}},
                                              "crop": {"zoom": 1.1}}
        p = self.fixture.root / "plan.json"
        fsutil.atomic_write_json(p, self.plan)
        options = dict(jobs=1, draft=True, encoder="cpu", asr=False, auto_calibrate=False)
        first = batch.run(p, **options)
        self.assertEqual(first["counts"]["rendered"], 1, first["reels"])
        effective = fsutil.read_json(batch.DATA / "batches" / self.plan["name"] / "reels/r1/style.effective.json")
        self.assertEqual(effective["captions"]["style"]["fill"], "#ffe600")
        second = batch.run(p, **options)
        self.assertEqual(second["counts"]["cached"], 1, second["reels"])


if __name__ == "__main__":unittest.main()
