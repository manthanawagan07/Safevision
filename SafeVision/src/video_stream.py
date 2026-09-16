"""
video_stream.py
----------------
Handles video-file and webcam input for the detection pipeline.

Designed to be headless-safe: frames are never displayed with
cv2.imshow() unless the user explicitly passes --display, so the
project runs correctly in a pure terminal / SSH environment
(Usability + Executability requirement).

Frame skipping (--every-n) keeps CPU usage bounded on long videos
(Performance + Resource Efficiency requirements).
"""

import os
from typing import Callable, Dict, List, Optional

import cv2

from src.detection import Detection, analyze_image, draw_annotations
from src.logger_setup import get_logger

logger = get_logger(__name__)


class VideoError(Exception):
    pass


def process_video(
    source,
    every_n: int = 15,
    max_frames: Optional[int] = None,
    display: bool = False,
    output_path: Optional[str] = None,
    on_frame: Optional[Callable[[int, List[Detection]], None]] = None,
) -> Dict:
    """
    Run the detection pipeline over a video file or webcam index.

    Parameters
    ----------
    source      : path to a video file, or an int webcam index (e.g. 0)
    every_n     : analyze only every Nth frame
    max_frames  : stop after this many analyzed frames (None = no limit)
    display     : show an annotated preview window (requires a GUI)
    output_path : if given, write an annotated .mp4 to this path
    on_frame    : optional callback invoked as on_frame(frame_index, detections)

    Returns a dict summarizing the run.
    """
    if isinstance(source, str) and not source.isdigit() and not os.path.exists(source):
        raise VideoError(f"Video file not found: {source}")

    capture_arg = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(capture_arg)

    if not cap.isOpened():
        raise VideoError(f"Could not open video source: {source}")

    writer = None
    frame_index = 0
    analyzed = 0
    totals = {"with_mask": 0, "without_mask": 0}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % every_n == 0:
                detections = analyze_image(frame)
                analyzed += 1

                for det in detections:
                    totals[det.label] = totals.get(det.label, 0) + 1

                if on_frame is not None:
                    on_frame(frame_index, detections)

                annotated = draw_annotations(frame, detections)

                if output_path is not None:
                    if writer is None:
                        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))
                    writer.write(annotated)

                if display:
                    cv2.imshow("SafeVision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User requested early exit from video stream.")
                        break

                if max_frames is not None and analyzed >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    summary = {
        "frames_read": frame_index,
        "frames_analyzed": analyzed,
        "with_mask": totals.get("with_mask", 0),
        "without_mask": totals.get("without_mask", 0),
    }
    logger.info("Video processing complete: %s", summary)
    return summary
