"""Job 2: the full "Toh Zinda Ho Tum" poem over a seamless forward-only ocean loop.

The loop (workspace/media/ocean_loop50.mp4) is the clip played forward with its end blended
into its start by optical-flow interpolation (see build_loop.sh); nothing is reversed."""
import json
import subprocess
import sys

D = 50.02
LINES = [  # (start, end, text): line timings found with Whisper on 2.5 s windows
    (0.6, 7.6, "Toh Zinda Ho Tum\n— Farhan Akhtar"),
    (7.8, 12.3, "Dilon mein tum apni\nbetaabiyan leke chal rahe ho"),
    (12.3, 13.8, "Toh zinda ho tum"),
    (13.8, 17.6, "Nazar mein khwabon ki\nbijliyan leke chal rahe ho"),
    (17.6, 19.9, "Toh zinda ho tum"),
    (19.9, 23.7, "Hawa ke jhokon ke jaise\naazad rehna seekho"),
    (23.7, 28.3, "Tum ek dariya ke jaise\nlehron mein behna seekho"),
    (28.3, 31.2, "Har ek lamhe se tum milo\nkhole apni baahein"),
    (31.2, 34.7, "Har ek pal ek naya samaa\ndekhen yeh nigaahein"),
    (34.7, 37.9, "Jo apni aankhon mein\nhairaniyan leke chal rahe ho"),
    (37.9, 40.3, "Toh zinda ho tum"),
    (40.3, 43.7, "Dilon mein tum apni\nbetaabiyan leke chal rahe ho"),
    (43.7, 48.5, "Toh zinda ho tum"),
]
STYLE = {"font": "Arimo", "weight": 700, "fontSize": 46, "color": "#ffffff", "bgColor": "#004B06", "bgOpacity": 1,
         "textShadow": 0, "strokeWidth": 0, "lineHeight": 1.3, "align": "center", "y": -360}  # sample reel's caption look

clips = [
    {"id": "v", "mediaId": "bg", "kind": "video", "track": "V1", "start": 0, "in": 0, "duration": D,
     "props": {"fit": "cover", "volume": 0}},
    {"id": "a", "mediaId": "poem", "kind": "audio", "track": "A1", "start": 0, "in": 0, "duration": D, "props": {}},
]
for i, (s, e, text) in enumerate(LINES):
    clips.append({"id": f"t{i}", "mediaId": None, "kind": "text", "track": "V3", "start": s, "duration": round(e - s, 3),
                  "props": dict(STYLE, text=text)})
project = {"name": "job2-poem", "width": 720, "height": 1280, "fps": 30, "revision": 1, "panSchema": 1,
           "background": "#000000",
           "media": [{"id": "bg", "name": "ocean_loop50.mp4", "kind": "video", "src": "/media/ocean_loop50.mp4"},
                     {"id": "poem", "name": "poem.flac", "kind": "audio", "src": "/media/poem.flac"}],
           "clips": clips}
json.dump(project, open("project.json", "w"), indent=1)
subprocess.run([sys.executable, "../../scripts/fast_render.py", "project.json", "out/job2-poem.mp4",
                "--cpu-preset", "veryfast"], check=True)
