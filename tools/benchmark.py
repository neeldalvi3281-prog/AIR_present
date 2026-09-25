"""Phase 1 benchmark: sweep resolutions headlessly, write JSON + print summary.

Measures: camera FPS (measured, never requested), inference FPS,
inference latency (mean/median/p95), end-to-end latency, frame drops,
hand detection rate, process CPU, RAM. No adaptive FPS/resolution,
no gestures (PRD Phase 1).

Modes:
  default            config resolutions @ target_fps (legacy behavior)
  --matrix           §5 resolution x requested-FPS grid (8 combos)
  --backend NAME     dshow | msmf | any (default: legacy dshow+fallback)
  --codec NAME       default | mjpg
  --seconds N / --warmup N
  --min-confidence F MediaPipe detection/tracking confidence (sweep §12)
  --write-profile    remember fastest proven mode -> camera_profile.json
  --list             print combos without touching the camera

Actual achieved width/height/FPS/codec are read back from the device
and reported (§8); requested values are never presented as results.
"""
import argparse
import json
import os
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.capture import CameraManager  # noqa: E402
from config import load_config  # noqa: E402
from performance.profiler import Profiler  # noqa: E402
from vision.features import derive  # noqa: E402
from vision.hand_tracker import HandTracker  # noqa: E402

# opti §5 capability grid: (width, height, requested_fps).
MATRIX = [(320, 240, 30), (320, 240, 60),
          (480, 270, 30), (480, 270, 60),
          (640, 360, 30), (640, 360, 60),
          (640, 480, 30), (640, 480, 60)]

PROFILE_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "camera_profile.json")


def run_resolution(width, height, seconds, warmup, tracker,
                   target_fps=30, backend=None, codec=None):
    cam = CameraManager(device=cfg_camera_device(),
                        width=width, height=height,
                        target_fps=target_fps,
                        backend=backend, codec=codec)
    cam.start()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    profiler = Profiler()
    last_seq = -1
    last_sysmon = 0.0
    cpu, ram = 0.0, 0.0

    def sample_sysmon():
        nonlocal last_sysmon, cpu, ram
        now = time.perf_counter()
        if now - last_sysmon >= 0.25:  # ~4 Hz, not per frame (§34)
            cpu = proc.cpu_percent(interval=None)
            ram = proc.memory_info().rss / (1024 * 1024)
            profiler.note_system(cpu, ram)
            last_sysmon = now

    # Warmup: let exposure/AE settle, fill pipeline (excluded from stats).
    t_end = time.perf_counter() + warmup
    while time.perf_counter() < t_end:
        frame, cap_ts, seq = cam.read_latest()
        if frame is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        _infer_once(frame, cap_ts, tracker, profiler=None)

    # Measured window.
    profiler = Profiler()
    last_seq = -1
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        frame, cap_ts, seq = cam.read_latest()
        if frame is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        now = time.perf_counter()
        profiler.note_capture(cap_ts)
        t0 = time.perf_counter()
        det = tracker.detect(frame, int(now * 1000))
        infer_s = time.perf_counter() - t0
        e2e_s = time.perf_counter() - cap_ts
        found = det is not None
        if found:
            h, w = frame.shape[:2]
            derive(det["landmarks_norm"], w, h)  # include feature cost
        profiler.note_inference(now, found, infer_s, e2e_s)
        sample_sysmon()
    stats = cam.stats()
    cam.stop()
    summary = profiler.summary(captured=stats["captured"],
                               overwritten_unread=stats["overwritten_unread"])
    summary["resolution"] = f"{width}x{height}"
    summary["width"] = width
    summary["height"] = height
    summary["requested_width"] = width
    summary["requested_height"] = height
    summary["requested_fps"] = target_fps
    summary["actual_width"] = stats["actual_width"]
    summary["actual_height"] = stats["actual_height"]
    summary["actual_fps"] = stats["actual_fps"]
    summary["backend"] = stats["backend"]
    summary["codec"] = stats["codec"]
    summary["seconds"] = seconds
    return summary


def _infer_once(frame, cap_ts, tracker, profiler):
    now = time.perf_counter()
    det = tracker.detect(frame, int(now * 1000))
    if profiler is not None:
        profiler.note_capture(cap_ts)
        profiler.note_inference(now, det is not None, 0.0, 0.0)
    return det


_CFG = None


def cfg_camera_device():
    return _CFG["camera"]["device"]


def cfg_camera_fps():
    return _CFG["camera"]["target_fps"]


def print_summary(s):
    il = s["infer_latency_ms"]
    el = s["e2e_latency_ms"]
    print("=== PRESENTER CONTROLLER BENCHMARK ===")
    print(f"Resolution: {s['resolution']}  "
          f"(actual {s.get('actual_width')}x{s.get('actual_height')} "
          f"@ {s.get('actual_fps')} fps, "
          f"{s.get('backend')}/{s.get('codec')})")
    print(f"Camera FPS: {s['camera_fps']}")
    print(f"Inference FPS: {s['inference_fps']}")
    print(f"Mean inference latency: {il['mean']:.2f} ms")
    print(f"Median inference latency: {il['median']:.2f} ms")
    print(f"P95 inference latency: {il['p95']:.2f} ms")
    print(f"End-to-end latency (mean): {el['mean']:.2f} ms")
    print(f"Frame drops: {s['frame_drops']} (rate {s['frame_drop_rate']})")
    print(f"Hand detection rate: {s['hand_detection_rate']}")
    print(f"CPU: mean {s['cpu_pct']['mean']:.1f}% p95 {s['cpu_pct']['p95']:.1f}%")
    print(f"RAM: mean {s['ram_mb']['mean']:.1f} MB max {s['ram_mb']['max']:.1f} MB")
    print("")


def write_profile(results, path=PROFILE_PATH):
    """Remember the fastest proven mode (highest measured inference FPS
    with a successfully opened camera). Only real modes count: rows
    where the device silently substituted another resolution are
    excluded. Manual command only (§48)."""
    viable = [r for r in results
              if r["camera_fps"] > 0
              and r.get("actual_width") == r.get("requested_width")
              and r.get("actual_height") == r.get("requested_height")]
    if not viable:
        print("No viable camera mode; profile not written.")
        return None
    best = max(viable, key=lambda r: r["inference_fps"])
    profile = {"backend": best["backend"], "codec": best["codec"],
               "width": best["actual_width"] or best["width"],
               "height": best["actual_height"] or best["height"],
               "fps": best["requested_fps"],
               "measured_inference_fps": best["inference_fps"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Wrote {path}: {profile}")
    return profile


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true",
                    help="run the resolution x FPS capability grid")
    ap.add_argument("--backend", default=None,
                    help="dshow | msmf | any (default: legacy behavior)")
    ap.add_argument("--codec", default=None,
                    help="default | mjpg (default: device default)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None)
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="MediaPipe confidence sweep value (§12)")
    ap.add_argument("--write-profile", action="store_true",
                    help="remember fastest proven mode")
    ap.add_argument("--list", action="store_true",
                    help="print combos without touching the camera")
    args = ap.parse_args(argv)

    global _CFG
    _CFG = load_config()
    model_path = _CFG["model"]["path"]
    bench = _CFG["benchmark"]
    seconds = args.seconds or bench["seconds_per_resolution"]
    warmup = args.warmup if args.warmup is not None else bench["warmup_seconds"]
    conf = (args.min_confidence
            or _CFG["tracking"]["min_detection_confidence"])
    if args.matrix:
        combos = [(w, h, fps) for w, h, fps in MATRIX]
    else:
        combos = [(w, h, cfg_camera_fps()) for w, h in bench["resolutions"]]
    if args.list:
        for w, h, fps in combos:
            print(f"{w}x{h} @ {fps}  backend={args.backend} codec={args.codec}")
        return []
    tracker = HandTracker(
        model_path,
        min_detection_confidence=conf,
        min_tracking_confidence=_CFG["tracking"]["min_tracking_confidence"],
    )
    results = []
    try:
        for w, h, fps in combos:
            print(f"Benchmarking {w}x{h} @ {fps} "
                  f"(backend={args.backend}, codec={args.codec}) "
                  f"for {seconds}s ...")
            try:
                s = run_resolution(w, h, seconds, warmup, tracker,
                                   target_fps=fps, backend=args.backend,
                                   codec=args.codec)
            except RuntimeError as e:
                print(f"SKIP {w}x{h} @ {fps}: {e}")
                continue
            results.append(s)
            print_summary(s)
    finally:
        tracker.close()
    out = {"model": {"path": model_path,
                     "mediapipe_version": __import__("mediapipe").__version__},
           "min_confidence": conf,
           "results": results}
    os.makedirs(os.path.dirname(bench["output"]) or ".", exist_ok=True)
    with open(bench["output"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {bench['output']}")
    if args.write_profile:
        write_profile(results)
    return results


if __name__ == "__main__":
    main()
