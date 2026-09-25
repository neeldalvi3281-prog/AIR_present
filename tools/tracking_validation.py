"""Phase 1.1 controlled tracking validation — measurement only.

Modes:
    python tools/tracking_validation.py --preview    # live window + guidance (for manual runs)
    python tools/tracking_validation.py --headless   # clean numbers, no drawing

Procedure per resolution (320x240, 480x270, 640x360):
  2 s warmup + 20 s measurement (overridable via --warmup/--seconds for smoke tests).

  "Keep one hand continuously visible in the center of the frame at
   approximately the same distance from the camera. Slowly move
   horizontally and vertically."

tracking_rate = frames_with_hand / processed_frames. It is ONLY a
resolution comparison when the hand really was continuously visible;
otherwise it measures hand presence, not resolution quality.

No gestures, control, pointer, laser, calibration, or adaptive logic.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.capture import CameraManager  # noqa: E402
from config import load_config  # noqa: E402
from performance.profiler import Profiler  # noqa: E402
from vision.features import derive  # noqa: E402
from vision.hand_tracker import HandTracker  # noqa: E402

RESOLUTIONS = [(320, 240), (480, 270), (640, 360)]
INSTRUCTION = ("Keep one hand continuously visible in the center of the frame "
               "at approximately the same distance from the camera. "
               "Slowly move horizontally and vertically.")


def run_resolution(width, height, tracker, device, target_fps,
                   warmup_s, run_s, preview=False):
    show = None
    if preview:
        import cv2
        from ui.developer import build_lines, draw
        show = (cv2, build_lines, draw)
    cam = CameraManager(device=device, width=width, height=height,
                        target_fps=target_fps)
    cam.start()
    proc = psutil.Process()
    psutil.cpu_percent(interval=None)
    confidences = []
    last_seq = -1
    ts_counter = [int(time.perf_counter() * 1000)]
    aborted = False

    def next_ts():
        now = int(time.perf_counter() * 1000)
        if now <= ts_counter[0]:
            now = ts_counter[0] + 1
        ts_counter[0] = now
        return now

    def pump(duration, profiler, label):
        nonlocal last_seq
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            frame, cap_ts, seq = cam.read_latest()
            if preview:
                cv2, _, _ = show
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    return False
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            if preview:
                frame = show[0].flip(frame, 1)  # selfie-view mirror, pre-detect
            now = time.perf_counter()
            if profiler is not None:
                profiler.note_capture(cap_ts)
            t0 = time.perf_counter()
            det = tracker.detect(frame, next_ts())
            infer_s = time.perf_counter() - t0
            if profiler is not None:
                found = det is not None
                profiler.note_inference(now, found, infer_s,
                                        time.perf_counter() - cap_ts)
                profiler.note_system(psutil.cpu_percent(interval=None),
                                     proc.memory_info().rss / (1024 * 1024))
                if found:
                    confidences.append(det["hand_confidence"])
                h, w = frame.shape[:2]
                feat = derive(det["landmarks_norm"], w, h) if found else None
                if preview:
                    cv2, build_lines, draw = show
                    s = profiler.summary()
                    remaining = max(0.0, end - time.perf_counter())
                    lines = [
                        "KEEP ONE HAND VISIBLE",
                        "Move slowly horizontally and vertically",
                        f"{label} {width}x{height}  {remaining:.0f}s left",
                        f"HAND: {'DETECTED' if found else 'NO HAND'}"
                        + (f" {confidences[-1]:.2f}" if found else ""),
                    ] + build_lines(found, confidences[-1] if found else 0.0,
                                    det["handedness"] if found else "",
                                    w, h, s["camera_fps"], s["inference_fps"],
                                    s["infer_latency_ms"]["mean"],
                                    s["infer_latency_ms"]["p95"],
                                    s["cpu_pct"]["mean"], s["ram_mb"]["mean"])
                    draw(frame, feat, lines)
                    cv2.imshow(f"Tracking validation {width}x{height} (Q/ESC exits)", frame)
            if preview:
                import cv2 as _cv
                if _cv.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    return False
        return True

    try:
        pump(warmup_s, None, "warmup")
        confidences.clear()
        profiler = Profiler()
        last_seq = -1
        ok = pump(run_s, profiler, "measure")
        aborted = not ok
        stats = cam.stats()
        s = profiler.summary(captured=stats["captured"],
                             overwritten_unread=stats["overwritten_unread"])
    finally:
        cam.stop()
        if preview:
            import cv2 as _cv
            _cv.destroyAllWindows()

    conf = np.asarray(confidences, dtype=np.float64) if confidences else np.array([])
    total = s["frames_seen"]
    with_hand = s["frames_with_hand"]
    row = {
        "resolution": f"{width}x{height}",
        "width": width, "height": height,
        "seconds": run_s, "warmup_s": warmup_s, "mode": "preview" if preview else "headless",
        "total_processed_frames": total,
        "frames_with_hand": with_hand,
        "frames_without_hand": total - with_hand,
        "tracking_rate": round(with_hand / total, 4) if total else 0.0,
        "avg_confidence": round(float(conf.mean()), 4) if conf.size else 0.0,
        "min_confidence": round(float(conf.min()), 4) if conf.size else 0.0,
        "confidence_p95": round(float(np.percentile(conf, 95)), 4) if conf.size else 0.0,
        "inference_mean_ms": round(s["infer_latency_ms"]["mean"], 2),
        "inference_median_ms": round(s["infer_latency_ms"]["median"], 2),
        "inference_p95_ms": round(s["infer_latency_ms"]["p95"], 2),
        "e2e_mean_ms": round(s["e2e_latency_ms"]["mean"], 2),
        "e2e_p95_ms": round(s["e2e_latency_ms"]["p95"], 2),
        "camera_fps": s["camera_fps"],
        "inference_fps": s["inference_fps"],
        "frame_drops": s["frame_drops"],
        "frame_drop_rate": s["frame_drop_rate"],
        "cpu_mean": round(s["cpu_pct"]["mean"], 1),
        "ram_mean_mb": round(s["ram_mb"]["mean"], 1),
        "aborted": aborted,
    }
    return row


def print_table(rows):
    hdr = ("Resolution", "Tracked", "TrackRate", "AvgConf", "MinConf", "P95Conf",
           "InfMean", "InfMed", "InfP95", "E2EMean", "E2EP95",
           "CamFPS", "InfFPS", "Drops", "CPU%", "RAMMB")
    print(" | ".join(hdr))
    print("-" * len(" | ".join(hdr)))
    for r in rows:
        print(f"{r['resolution']:>10} | {r['frames_with_hand']:>7}/{r['total_processed_frames']:<5} | "
              f"{r['tracking_rate']:>9} | {r['avg_confidence']:>7} | {r['min_confidence']:>7} | "
              f"{r['confidence_p95']:>7} | {r['inference_mean_ms']:>7} | {r['inference_median_ms']:>6} | "
              f"{r['inference_p95_ms']:>6} | {r['e2e_mean_ms']:>7} | {r['e2e_p95_ms']:>6} | "
              f"{r['camera_fps']:>6} | {r['inference_fps']:>6} | {r['frame_drops']:>5} | "
              f"{r['cpu_mean']:>4} | {r['ram_mean_mb']:>6}")


def summarize_best(rows):
    """Name the best measured value per category. No weighting, no ranking."""
    def best(key, lower_is_better=True):
        cands = [r for r in rows if r["total_processed_frames"] > 0]
        if not cands:
            return None
        return min(cands, key=lambda r: r[key]) if lower_is_better else max(cands, key=lambda r: r[key])
    b_track = best("tracking_rate", lower_is_better=False)
    b_e2e = best("e2e_mean_ms", lower_is_better=True)
    b_drop = best("frame_drop_rate", lower_is_better=True)
    b_cpu = best("cpu_mean", lower_is_better=True)
    print("\n--- Best measured per category (raw values, no weighting/ranking) ---")
    if b_track: print(f"tracking rate : {b_track['resolution']} ({b_track['tracking_rate']})")
    if b_e2e: print(f"E2E latency   : {b_e2e['resolution']} (mean {b_e2e['e2e_mean_ms']} ms)")
    if b_drop: print(f"frame drops   : {b_drop['resolution']} ({b_drop['frame_drops']} drops, rate {b_drop['frame_drop_rate']})")
    if b_cpu: print(f"CPU usage     : {b_cpu['resolution']} (mean {b_cpu['cpu_mean']}%)")
    print("NOTE: tracking_rate is only a resolution comparison if the hand was "
          "continuously visible during every run.")


def mean_rows(all_runs):
    """Mean across --runs per resolution (individual runs preserved in JSON)."""
    agg = {}
    for r in all_runs:
        agg.setdefault(r["resolution"], []).append(r)
    means = []
    for res, rs in agg.items():
        m = {"resolution": res, "runs": len(rs)}
        for k in ("tracking_rate", "avg_confidence", "inference_mean_ms",
                  "inference_median_ms", "inference_p95_ms", "e2e_mean_ms",
                  "e2e_p95_ms", "camera_fps", "inference_fps",
                  "frame_drop_rate", "cpu_mean", "ram_mean_mb"):
            m["mean_" + k] = round(sum(x[k] for x in rs) / len(rs), 4)
        means.append(m)
    return means


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preview", action="store_true", help="live window with guidance")
    g.add_argument("--headless", action="store_true", help="clean numbers, no drawing")
    ap.add_argument("--runs", type=int, default=1, help="repeat each resolution N times")
    ap.add_argument("--seconds", type=float, default=20.0, help="measurement seconds")
    ap.add_argument("--warmup", type=float, default=2.0, help="warmup seconds")
    args = ap.parse_args()
    preview = args.preview or (not args.headless and False)
    # Default to headless when neither flag given (clean numbers).
    if not args.preview and not args.headless:
        preview = False
    cfg = load_config()
    tracker = HandTracker(cfg["model"]["path"])
    all_runs = []
    try:
        for run in range(args.runs):
            for i, (w, h) in enumerate(RESOLUTIONS):
                print(f"[run {run + 1}/{args.runs} res {i + 1}/{len(RESOLUTIONS)}] {w}x{h} "
                      f"({'preview' if preview else 'headless'}): {INSTRUCTION} "
                      f"Warmup {args.warmup}s + measure {args.seconds}s ...")
                row = run_resolution(w, h, tracker, cfg["camera"]["device"],
                                     cfg["camera"]["target_fps"],
                                     args.warmup, args.seconds, preview=preview)
                row["run"] = run + 1
                all_runs.append(row)
                print(f"  -> with {row['frames_with_hand']}/without {row['frames_without_hand']} "
                      f"(rate {row['tracking_rate']}, avg conf {row['avg_confidence']})")
                if row["aborted"]:
                    print("  aborted by user (Q/ESC).")
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        tracker.close()
    # Latest full-resolution set (last run) drives the table; means cover --runs.
    latest = all_runs[-len(RESOLUTIONS):] if all_runs else []
    out = {"model": {"path": cfg["model"]["path"],
                     "mediapipe_version": __import__("mediapipe").__version__},
           "procedure": {"warmup_s": args.warmup, "run_s": args.seconds,
                         "runs": args.runs, "mode": "preview" if preview else "headless",
                         "instruction": INSTRUCTION,
                         "validity_note": "tracking_rate compares resolutions ONLY if the hand was "
                                          "continuously visible in every run; otherwise it measures presence."},
           "runs": all_runs,
           "mean_across_runs": mean_rows(all_runs) if args.runs > 1 else [],
           "results": latest}
    os.makedirs("logs", exist_ok=True)
    with open("logs/tracking_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n=== TRACKING VALIDATION COMPARISON ===")
    print_table(latest)
    summarize_best(latest)
    if args.runs > 1:
        print("\n--- Mean across runs ---")
        for m in out["mean_across_runs"]:
            print(m)
    print("\nWrote logs/tracking_validation.json")


if __name__ == "__main__":
    main()
