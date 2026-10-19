"""reelkit: sample reel + long-form footage in, finished reels out.

Pipeline (each stage cached on disk, keyed by content):

    ingest  → asset store (sha-256 dedupe)                         store.py
    sample  → measured style spec ("reel DNA")                     sample.py, grade.py
    source  → audio, transcript, sentence index                    source.py, asr.py
    plan    → which sentences make which reel (written by Claude)  plan.py
    compile → timeline per reel (cuts, reframing, captions, …)     compile.py
    render  → compositor → ffmpeg encode + audio graph → mp4       render.py, captions.py
    qc      → technical, edit and style checks + review sheet      qc.py
    batch   → all of the above for N reels in parallel             batch.py
"""

__version__ = "2.0.0"
