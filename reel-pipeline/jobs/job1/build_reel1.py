"""Builds reel 1 (raw R1 → sample S1 style) and PUTs it to the FableCut server."""
import json
import urllib.request

API = "http://localhost:7777/api/project"
W, H = 1080, 1920

# source windows (raw seconds), punch-in, captions in source time
WIDE = dict(s=1.0, x=0, y=0)
CELLS = dict(s=1.5, x=60, y=-96)        # grid around C8
COLS = dict(s=1.8, x=150, y=375)        # column header band above the caption
ROWS = dict(s=1.8, x=432, y=0)          # row numbers, left edge
E7 = dict(s=2.0, x=-540, y=-120)        # cell E7 above the caption
SEGS = [
    (0.20, 4.25, WIDE,  [(0.20, 2.20, "Excel Basics EVERY\nBeginner Must Know!!"), (2.20, 4.25, "Open a Blank Workbook")]),
    (4.35, 9.40, WIDE,  [(4.35, 9.40, "See all these rectangles?")]),
    (9.78, 12.15, CELLS, [(9.78, 12.15, "These are CELLS")]),
    (12.32, 15.20, COLS, [(12.32, 15.20, "Letters across the top...")]),
    (15.45, 17.15, COLS, [(15.45, 17.15, "...are COLUMNS")]),
    (17.30, 21.05, ROWS, [(17.30, 21.05, "Numbers down the side...")]),
    (21.20, 23.15, ROWS, [(21.20, 23.15, "...are ROWS")]),
    (23.45, 28.38, E7,  [(23.45, 26.95, "Column + Row..."), (26.95, 28.38, "= Cell E7!")]),
    (28.38, 29.95, E7,  [(28.38, 29.95, "Send this to a friend\nwho's new to Excel!!")]),
]
MUTED = {8}   # last segment: narration is cut off mid-sentence

CAPTION = {"font": "Arimo", "weight": 700, "fontSize": 56, "color": "#ffffff",
           "bgColor": "#004B06", "bgOpacity": 1, "textShadow": 0, "strokeWidth": 0,
           "lineHeight": 1.25, "align": "center", "y": 80}

media = [{"id": "m_r1", "name": "R1.edit.mp4", "kind": "video", "src": "/media/R1.edit.mp4"}]
clips, t = [], 0.0
for i, (a, b, z, caps) in enumerate(SEGS):
    d = round(b - a, 3)
    props = {"fit": "cover", "scale": z["s"], "x": z["x"], "y": z["y"], "volume": 0}  # sound comes from the A1 clips
    clips.append({"id": f"v{i}", "mediaId": "m_r1", "kind": "video", "track": "V1", "start": round(t, 3),
                  "in": a, "duration": d, "linkGroup": f"lg{i}", "props": props})
    clips.append({"id": f"a{i}", "mediaId": "m_r1", "kind": "audio", "track": "A1", "start": round(t, 3),
                  "in": a, "duration": d, "linkGroup": f"lg{i}",
                  "props": {"volume": 0 if i in MUTED else 1, "pan": 0, "channelMode": "mono"}})
    for j, (ca, cb, text) in enumerate(caps):
        clips.append({"id": f"t{i}_{j}", "mediaId": None, "kind": "text", "track": "V3",
                      "start": round(t + ca - a, 3), "duration": round(cb - ca, 3),
                      "props": dict(CAPTION, text=text)})
    t += d

cur = json.load(urllib.request.urlopen(API))
proj = {"name": "job1-reel1", "width": W, "height": H, "fps": 30, "revision": cur.get("revision", 0) + 1,
        "panSchema": 1, "background": "#ffffff", "media": media, "clips": clips}
json.dump(proj, open("project-1.json", "w"), indent=1)
req = urllib.request.Request(API, data=json.dumps(proj).encode(), method="PUT", headers={"content-type": "application/json"})
print(urllib.request.urlopen(req).read().decode(), f"length {t:.2f}s, {len(clips)} clips")
