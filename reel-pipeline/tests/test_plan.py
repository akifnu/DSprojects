"""plan.py: validation, word selection, fillers/stumbles/drops, jump cuts, silence snapping, overrides."""
import unittest

import numpy as np

from reelkit import asr, plan

WORDS = [("Hello", 0.5, 0.9), ("everyone.", 1.0, 1.5), ("Um", 1.8, 2.0), ("this", 2.2, 2.4), ("is", 2.5, 2.7),
         ("the", 2.8, 3.0), ("the", 3.1, 3.3), ("best", 3.4, 3.8), ("trick.", 3.9, 4.4),
         ("Watch", 5.6, 6.0), ("closely.", 6.1, 6.7)]


def transcript():
    words = [{"w": w, "s": s, "e": e} for w, s, e in WORDS]
    return {"lang": "en", "timing": "word", "words": words, "sentences": asr.sentences(words)}


def envelope(tr, quiet_at=()):
    """Loud inside words, near-silent between them (5 ms hop); optional extra dips."""
    env = np.full(int(8 / plan.HOP), 1e-4, np.float32)
    for w in tr["words"]:
        env[int(w["s"] / plan.HOP):int(w["e"] / plan.HOP)] = 0.3
    for t in quiet_at:
        env[int(t / plan.HOP)] = 1e-6
    return env


class Plan(unittest.TestCase):
    def test_validation(self):
        good = {"style": "x", "sources": {"S": "a"}, "reels": [{"id": "r1", "segments": [{"src": "S", "s": 0, "e": 1}]}]}
        self.assertEqual(plan.load(good)["reels"][0]["id"], "r1")
        for bad in ({"style": "x", "sources": {"S": "a"}, "reels": [{"id": "r1", "segments": []}]},
                    {"style": "x", "sources": {"S": "a"}, "reels": [{"id": "r1", "segments": [{"src": "T", "s": 0, "e": 1}]}]},
                    {"style": "x", "sources": {"S": "a"}, "reels": [{"id": "r", "segments": [{"src": "S", "s": 0, "e": 1}]},
                                                                    {"id": "r", "segments": [{"src": "S", "s": 0, "e": 1}]}]}):
            with self.assertRaises(ValueError):
                plan.load(bad)

    def test_sentences_and_cleaning(self):
        tr = transcript()
        self.assertEqual([s["text"] for s in tr["sentences"]],
                         ["Hello everyone.", "Um this is the the best trick.", "Watch closely."])
        idx = plan.segment_words(tr, {"sentences": [0, 1]})
        kept, removed = plan.clean(tr, idx)
        self.assertEqual([tr["words"][i]["w"] for i in kept],
                         ["Hello", "everyone.", "this", "is", "the", "best", "trick."])
        self.assertEqual(removed, {2: "filler", 5: "stumble"})
        kept, removed = plan.clean(tr, idx, drop=[[1, 2]])      # sentence 1, word 2 = "is"
        self.assertEqual(removed[4], "dropped")

    def test_runs_cut_at_removals_and_long_pauses_and_snap_into_silence(self):
        tr = transcript()
        env = envelope(tr, quiet_at=[0.47, 1.56, 2.13])
        reel = {"id": "r", "segments": [{"src": "S", "sentences": [0, 2]}]}
        runs = plan.resolve_reel(reel, {"S": tr}, {"S": env}, max_pause=0.4)
        texts = [" ".join(w["w"] for w in r["words"]) for r in runs]
        self.assertEqual(texts, ["Hello everyone.", "this is", "the best trick.", "Watch closely."])
        r0 = runs[0]
        self.assertAlmostEqual(r0["s"], 0.47, delta=0.006)        # the quietest point before "Hello"
        self.assertAlmostEqual(r0["e"], 1.56, delta=0.006)        # ... and after "everyone."
        self.assertAlmostEqual(runs[1]["s"], 2.13, delta=0.006)
        for r in runs:                                            # never inside a word
            for w in tr["words"]:
                self.assertFalse(w["s"] < r["s"] < w["e"] or w["s"] < r["e"] < w["e"], (r, w))
            self.assertLessEqual(r["s"], r["words"][0]["s"])
            self.assertGreaterEqual(r["e"], r["words"][-1]["e"])

    def test_short_pauses_stay_in_one_run(self):
        tr = transcript()
        runs = plan.resolve_reel({"id": "r", "segments": [{"src": "S", "sentences": [2, 2]}]}, {"S": tr},
                                 {"S": envelope(tr)}, max_pause=0.4)
        self.assertEqual(len(runs), 1)

    def test_hook_first_and_caption_overrides(self):
        tr = transcript()
        reel = {"id": "r", "hook": {"src": "S", "sentences": [2]}, "segments": [{"src": "S", "sentences": [0, 0]}],
                "caption_text": {"0": "Hi everybody!", "2": "Watch very closely."}}
        runs = plan.resolve_reel(reel, {"S": tr}, {"S": envelope(tr)}, max_pause=0.4)
        self.assertTrue(runs[0]["hook"])
        self.assertEqual([w["w"] for w in runs[0]["words"]], ["Watch", "very", "closely."])
        self.assertEqual([w["w"] for w in runs[1]["words"]], ["Hi", "everybody!"])
        ws = runs[0]["words"]
        self.assertAlmostEqual(ws[0]["s"], 5.6, delta=0.01)
        self.assertAlmostEqual(ws[-1]["e"], 6.7, delta=0.01)
        self.assertTrue(all(a["e"] <= b["s"] + 1e-6 for a, b in zip(ws, ws[1:])))

    def test_explicit_times(self):
        tr = transcript()
        runs = plan.resolve_reel({"id": "r", "segments": [{"src": "S", "s": 5.5, "e": 6.8}]}, {"S": tr},
                                 {"S": envelope(tr)}, max_pause=0.4)
        self.assertEqual([w["w"] for w in runs[0]["words"]], ["Watch", "closely."])


if __name__ == "__main__":
    unittest.main()
