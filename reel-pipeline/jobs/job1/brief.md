# job1 — brief

Sample S1 (`Video-24737.mp4`; the three uploads were byte-identical, so one sample).
An Excel tip reel ("DON'T Stretch Columns like a NOOB!!", @ExcelbyChris style).

## Sample S1
| length | canvas | detected shots | avg shot | BPM | drop |
|---|---|---|---|---|---|
| 25.7 s | 720×1280 @ 30 (deliver 1080×1920) | 4 (cuts 1.97 · 9.27 · 10.47) | 6.4 s | ~96 (clicks/typing, not music) | 21.5 |

Picture cuts are few on purpose. The rhythm comes from the **captions, which change
every 1.5–2 s**, and from punch-ins on the part of the screen being used.

**Format:** a screen recording of one task, filled edge to edge in 9:16. Wide (full UI)
for context, then a **punch-in (~1.5–2×)** on the cell, menu or code being touched.
Hard cuts only, no transitions. Typing and menu steps are sped up, and dead time is cut.

**Hook (0–2 s):** a bold claim / "you're doing it wrong" caption over the first frame of
the problem: *DON'T Stretch Columns like a NOOB!!*

**Structure (beats):**
| t | caption | picture |
|---|---|---|
| 0.0–2.0 | DON'T Stretch Columns like a NOOB!! | the problem (empty sheet) |
| 2.0–4.4 | — (no caption) | the slow way: dragging column widths (punch-in) |
| 4.4–5.4 | Instead… | |
| 5.5–6.4 | Try this! | |
| 6.5–7.4 | Right click the… | sheet tab, zoomed |
| 7.5–8.0 | Worksheet Tab | context menu |
| 8.0–9.6 | Select "View Code" | |
| 10.0–10.5 | — | VBA editor opens (cut) |
| 11.0–12.6 | Change "General" to "Worksheet" | dropdown, zoomed |
| 13.5–15.4 | Now enter this secret code | code being typed |
| 15.5–17.9 | — | code finished |
| 18.0–18.4 | That's it! | back to the sheet (cut) |
| 18.5–20.4 | Every column Auto-fits | typing headers, columns grow |
| 20.5–21.4 | to the data! | payoff |
| 22.0–24.4 | Send this video to your boss to get a raise!! | CTA over the payoff |
| 25.0–25.7 | — | end / loop |

**Captions:** sentence case with CAPS for emphasis; each one restates the voice-over line.
White bold sans (Arial/Helvetica-bold look → `font:"Inter"`, weight 700), solid **dark-green
rounded pill `#004B06`, opacity 1**, no stroke, no shadow. About 34 px on 720 w
(→ **≈ 50 px at 1080 w**), centred, just below the middle of the frame (**y ≈ +80 px
at 1080×1920**). Up to 2 lines, ~22 characters per line. No caption animation:
captions hard-cut on and off with the phrase.
**Look:** no grade, raw UI capture, white UI background.
**Audio:** voice-over narration (one short line per caption) + UI click/typing
sounds, no music. **−14.1 LUFS** integrated.
**End card / CTA:** a share CTA caption ("Send this video to your boss…") over the result,
no separate end card.

## Template for new reels
1. Hook claim caption (2 s) over the problem.
2. Show the slow / wrong way (2–3 s).
3. "Instead… / Try this!"
4. 3–5 steps, one caption + punch-in each (1.5–2.5 s per step).
5. "That's it!" → payoff shot with the result caption.
6. Share CTA caption over the result (~2.5 s). Total 20–30 s.

## Raw footage
R1 (`VID-20260925-WA0012.mp4`): 30 s, 2160×3840 @ 60, HEVC, already 9:16. Excel basics
screen recording (open a blank workbook → cells → columns → rows → cell E7) with
voice-over at −24.7 LUFS. The narration is cut off mid-sentence at 29.9 s. The opening
start-screen transition is smeared in the source (low-bitrate recompression).
Transcript: `work/R1.transcript.json`.

## Selects (reel 1; times in R1)
| in → out | VO | punch-in | caption |
|---|---|---|---|
| 0.20 → 4.25 | "New workbook. This drops us into a brand new workbook." | wide | hook "Excel Basics EVERY Beginner Must Know!!" → "Open a Blank Workbook" |
| 4.35 → 9.40 | "…a lot of rectangles on the screen." | wide | See all these rectangles? |
| 9.78 → 12.15 | "These are all referred to as cells." | 1.5× grid | These are CELLS |
| 12.32 → 15.20 | "Across the top… letters." | 1.8× headers | Letters across the top... |
| 15.45 → 17.15 | "These are referred to as columns." | 1.8× headers | ...are COLUMNS |
| 17.30 → 21.05 | "…numbers going down the screen." | 1.8× row numbers | Numbers down the side... |
| 21.20 → 23.15 | "These are referred to as rows." | 1.8× row numbers | ...are ROWS |
| 23.45 → 28.38 | "The intersection of the columns, column and the row… cell E7." (kept whole: a cut on the stumble removed "of the column") | 2× E7 | Column + Row... → = Cell E7! |
| 28.38 → 29.95 (muted) | cut-off sentence dropped | 2× E7 | Send this to a friend who's new to Excel!! (share CTA, like the sample's) |

## Reels
| # | length | output | notes / approximations |
|---|---|---|---|
| 1 | 28.25 s | `out/job1-reel1_reel_1080x1920.mp4` · `project-1.json` · `build_reel1.py` | caption font Arimo (Arial-metric) for the sample's Arial-bold look; no click SFX added (the source has its own) |
