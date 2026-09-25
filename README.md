# AIR_present — webcam hand-gesture presenter control (Windows, Python 3.12)

Move the cursor, turn slides, click, drag, zoom, and scroll — with one hand
in front of a webcam. MediaPipe Hand Landmarker (`hand_landmarker.task`,
float16/1, ~7.8 MB) supplies 21 landmarks per frame; all gesture logic is
deterministic landmark math, and all input goes through native Windows
`SendInput` (no automation libraries).

## Gestures

| Hand | Action |
|---|---|
| Palm (move anywhere) | Mouse cursor (whole-hand centroid) |
| Open palm, rotate clockwise | Previous slide (Left Arrow) |
| Open palm, rotate anti-clockwise | Next slide (Right Arrow) |
| Pinch (thumb + index, hold) | Left click / hold / drag |
| YO sign (index + pinky) + twist CW / ACW | Zoom in / out (Ctrl+wheel) |
| Peace sign pointed right / left | Scroll up / down |
| Fist | Nothing (resets sessions) |
| Hand gone | Everything stops; resume is jump-free |

## Quickstart

```bash
pip install -r requirements.txt
python tools/download_model.py   # fetches hand_landmarker.task -> models/
python app.py --preview          # dry-run overlay (Q/ESC quit, P overlay)
python -m unittest discover tests -v
```

## Going live

1. Verify the dry-run overlay first (nothing is sent).
2. Focus your slideshow window (keys/wheel go to the focused window).
3. In `config.json` → `action`: `enabled: true`, `dry_run: false`.
   (This repo's config ships live-enabled.)
4. To stop: set `enabled: false` again, or quit with Q/ESC
   (held buttons/keys are released on shutdown).

## Calibration

The orange `MOUSE CONTROL AREA` box is the entire mouse range — put your
palm at its corners to reach the screen corners. The `Palm:` / `Screen:`
readout lines show live coordinates. Tune the four numbers in
`config.json` → `[mouse.mapping]` (`left`/`right`/`top`/`bottom`); other
tunables (`rotation`, `pinch`, `zoom`, `scroll`, `fist`) live beside them.

## Benchmarks

```bash
python tools/benchmark.py                                   # default run
python tools/benchmark.py --matrix --seconds 6 --warmup 1   # capability grid
python tools/benchmark.py --matrix --write-profile          # fastest proven mode
```
Options: `--backend {dshow,msmf,any}`, `--codec {default,mjpg}`,
`--min-confidence F`, `--list`. Results → `logs/benchmark_latest.json`;
only actually measured FPS is ever reported.

## Project layout

- `vision/` — poses, pinch, palm rotation, YO zoom, peace scroll, shared geometry
- `actions/` — dispatcher + native SendInput adapters (mouse, wheel, keyboard)
- `camera/` — latest-frame single-slot capture thread
- `ui/` — developer preview overlay
- `tools/` — model download, benchmark, tracking validation
- `tests/` — deterministic suites (no webcam)
- `performance/` — rolling profiler (cached summaries)

## Safety notes

- Dry-run/observe modes never emit input; LIVE requires both
  `enabled: true` **and** `dry_run: false`.
- Hand loss releases a held button exactly once; shutdown releases
  everything; Ctrl is always released after each zoom tick.

## Reform — whole-hand mouse + palm rotation + pinch click

Interaction model (no modes, no swipe, no lock, no F8/F9):

* **Mouse (always):** palm centroid (wrist + 4 MCPs) → One-Euro filter →
  fixed virtual control box (`left` 0.07, `right` 0.95, `top` 0.08,
  `bottom` 0.92) → native absolute `SendInput`. Never the index tip
  alone. The box is static (never follows the hand); its edges map to
  the screen edges with outside clamping, so all four corners are
  reachable. The preview draws the box (`MOUSE CONTROL AREA`, orange)
  separately from the hand bbox, plus a palm dot and Palm/Screen
  readout lines for calibration. One-Euro handles edge stability — no
  large dead zone. Hand orientation does not matter: horizontal,
  diagonal, or upside-down hands track the same (centroid is
  rotation-invariant). Hand loss stops movement (no recenter jump).
* **Previous slide:** with a **wide-open palm only** (FIST/UNKNOWN/pinch-held
  reset the session), physical palm rotation CLOCKWISE in 30° steps
  (MCP-centroid axis + light angle smoothing; each step confirmed over
  4 frames; committed angle advances per step in landmark space, so
  continuous rotation pages continuously with no return-to-neutral;
  120 ms spacing between steps) = Left Arrow.
  The hand does **not** need to be upright: the session commits whatever
  angle the palm already has and every +30° of *relative* rotation pages
  once. Holding still after a step never repeats; CW/ACW cannot flip from
  landmark noise alone.
* **Next slide:** ANTI-CLOCKWISE 30° steps, same gating = Right Arrow.
  Reversing direction consumes steps back naturally. Non-open poses
  never navigate; hand loss resets the rotation session.
  Rotation never clicks.
* **Click (hold / drag):** **pinch only** — `||index tip − thumb tip|| /
  hand_scale` (wrist→middle MCP), lightly smoothed. Start threshold 0.30,
  release 0.38 (hysteresis: enter at start, exit only at release — jitter
  inside the band never cancels), 3-frame confirm → LEFT BUTTON DOWN;
  release or hand loss → exactly one LEFT BUTTON UP. No OPEN-pose
  requirement (other fingers may fold). Palm keeps steering while held,
  so drag works. 600 ms re-press cooldown; shutdown always releases.
  Pinch never dispatches while the YO sign holds. Fist does **nothing**.
* **Zoom (YO sign + twist):** hold YO (index + pinky out, middle + ring
  folded, thumb folded) and rotate — every +30° clockwise = one
  Ctrl+wheel-forward tick (zoom in), every −30° = one tick back
  (committed anchor advances per step so continuous twisting keeps
  zooming; 4-frame confirm, hysteresis, 250 ms tick spacing). Mouse
  keeps steering (zoom lands at the cursor). While YO holds, page nav
  and pinch clicks are silent; any other pose or hand loss resets the
  zoom anchor with no stale steps.
* **Scroll (peace sign):** hold PEACE (index + middle out, ring + pinky
  folded, thumb folded) pointed right to scroll up, left to scroll down
  (plain wheel ticks on the focused window). 4-frame entry confirm, then
  one tick every 200 ms while held; direction flip re-confirms; anything
  else resets silently. While PEACE holds, page nav and pinch clicks are
  silent.
* Hand gone → no move/click/nav; if pinched, one UP first; reappearance
  resumes clean, no jumps.
* **Reference lifetime:** commits **once** on wide-open palm; FIST,
  UNKNOWN, pinch-held, or hand loss **reset immediately**; next open is a
  fresh baseline (a stationary fist at any angle never navigates).
* Poses classified: OPEN / FIST / YO / PEACE / UNKNOWN (rotation gate
  + overlay). All geometry stays orientation-independent (YO holds while
  twisting — ratios don't change under rotation).
* No upright-hand requirement anywhere: orientation 0° is only a *label*
  for "fingers image-up" in the angle readout, never a gate.

Manual test (mirrored selfie view + palm/orientation/rotation/pinch panels):
`python app.py --preview` (Q/ESC quits, P toggles overlay)

Deterministic tests (no webcam): `python -m unittest discover tests -v`

Tunables live in `config.json` → `mouse` (control box `left`/`right`/
`top`/`bottom` 0.07/0.95/0.08/0.92, filter), `rotation`
(step_degrees 30, confirm 4, hysteresis 3, step spacing 120 ms,
open_required), `pinch` (start 0.30, release 0.38, confirm 3,
cooldown 600 ms), `zoom` (step_degrees 30, confirm 4, hysteresis 3,
tick spacing 250 ms), `scroll` (confirm 4, repeat 200 ms),
`fist` (fold_ratio / folded_min for pose only).

Dry run (safe, default — nothing is sent):
`python app.py --preview`
Shows GESTURE / ACTION / INPUT / MODE: DRY_RUN-or-OBSERVED.

Live control (explicit only):
1. Verify dry-run mapping first.
2. In `config.json` → `action`: set `enabled: true`, `dry_run: false`.
3. Open the presentation (slideshow mode) and make it the focused window —
   keys go to whichever window is focused; no auto-detection.
4. To disable: set `enabled: false` again (or close with Q/ESC;
   held keys are released on shutdown).
