#!/usr/bin/env python3
"""
main.py
-------
SafeVision command-line interface.

This is the single entry point through which a user interacts with the
system. Every feature is reachable from a terminal, with no GUI needed.

Workflow:
    1. `init`                     -> create DB tables + default admin
    2. `register` / (login flags) -> user management
    3. `detect-image` / `detect-video` -> run the CV pipeline, persist results
    4. `records`                  -> list / update / delete stored detections
    5. `report`                   -> generate compliance analytics

Run `python main.py --help` or `python main.py <command> --help`.
"""

import argparse
import getpass
import os
import sys
import time

import config
from src import auth, records, analytics
from src.logger_setup import get_logger

logger = get_logger("main")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_AUTH = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bootstrap_database() -> None:
    auth.init_auth_table()
    records.init_records_table()


def _authenticate(args) -> auth.User:
    """Resolve credentials from flags, env vars, or an interactive prompt."""
    username = args.username or os.environ.get("SAFEVISION_USER")
    password = args.password or os.environ.get("SAFEVISION_PASS")

    if not username:
        username = input("Username: ").strip()
    if not password:
        password = getpass.getpass("Password: ")

    return auth.login(username, password)


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_init(args) -> int:
    _bootstrap_database()
    auth.ensure_default_admin()

    from src.utils import save_sample_images
    save_sample_images(n=6)

    _print_header("SafeVision initialized")
    print(f"Database : {config.DATABASE_PATH}")
    print(f"Samples  : {config.SAMPLE_IMAGES_DIR}")
    print(f"Logs     : {config.LOG_FILE}")
    print("\nDefault admin account -> username: admin | password: admin123")
    print("Next step: python main.py detect-image --path data/sample_images")
    return EXIT_OK


def cmd_register(args) -> int:
    _bootstrap_database()
    password = args.new_password or getpass.getpass("Password for new user: ")
    auth.register_user(args.new_username, password, role=args.role)
    print(f"User '{args.new_username}' registered with role '{args.role}'.")
    return EXIT_OK


def cmd_detect_image(args) -> int:
    _bootstrap_database()
    user = _authenticate(args)

    # Imported here so that lightweight commands don't pay the CV import cost.
    import cv2
    from src.detection import analyze_image, save_annotated_image
    from src.preprocessing import list_images_in_dir, load_image, PreprocessingError

    if os.path.isdir(args.path):
        paths = list_images_in_dir(args.path)
    else:
        paths = [args.path]

    if not paths:
        print(f"No supported images found at: {args.path}")
        return EXIT_ERROR

    _print_header(f"Analyzing {len(paths)} image(s)")
    total_faces = 0
    processed = 0
    start = time.perf_counter()

    for path in paths:
        try:
            image = load_image(path)
        except PreprocessingError as exc:
            print(f"  [SKIP] {os.path.basename(path)}: {exc}")
            logger.error("Skipping unreadable file %s: %s", path, exc)
            continue

        detections = analyze_image(image)
        total_faces += len(detections)
        processed += 1

        n_real = sum(1 for d in detections if not d.is_fallback)
        n_fallback = sum(1 for d in detections if d.is_fallback)
        header = f"{n_real} face(s)"
        if n_fallback:
            header += f" (+{n_fallback} whole-image fallback)"
        print(f"\n  {os.path.basename(path)} -> {header}")

        for det in detections:
            flag = "OK " if det.label == "with_mask" else "!! "
            note = "  [whole-image]" if det.is_fallback else ""
            print(f"    {flag}{det.label:<14} conf={det.confidence:.2f}  box={det.box}{note}")

        records.bulk_create_records(os.path.basename(path), detections, user.username)

        if args.save_annotated:
            out = os.path.join(config.REPORTS_DIR, "annotated", os.path.basename(path))
            save_annotated_image(image, detections, out)

    elapsed = time.perf_counter() - start

    if processed == 0:
        print("\nNo image could be read. Check the --path argument.")
        return EXIT_ERROR

    print(f"\nDone. {total_faces} detection(s) across {processed} image(s) in {elapsed:.2f}s.")
    if args.save_annotated:
        print(f"Annotated images written to {os.path.join(config.REPORTS_DIR, 'annotated')}")
    return EXIT_OK


def cmd_detect_video(args) -> int:
    _bootstrap_database()
    user = _authenticate(args)

    from src.video_stream import process_video

    def _persist(frame_index, detections):
        source_label = f"{os.path.basename(str(args.source))}#frame{frame_index}"
        records.bulk_create_records(source_label, detections, user.username)

    _print_header(f"Processing video source: {args.source}")
    summary = process_video(
        source=args.source,
        every_n=args.every_n,
        max_frames=args.max_frames,
        display=args.display,
        output_path=args.output,
        on_frame=_persist,
    )

    for key, value in summary.items():
        print(f"  {key:<16}: {value}")
    return EXIT_OK


def cmd_records(args) -> int:
    _bootstrap_database()
    user = _authenticate(args)

    if args.delete is not None:
        auth.require_admin(user)
        ok = records.delete_record(args.delete)
        print(f"Record {args.delete} {'deleted.' if ok else 'not found.'}")
        return EXIT_OK if ok else EXIT_ERROR

    if args.update is not None:
        if not args.label:
            print("--label is required together with --update.")
            return EXIT_ERROR
        ok = records.update_record_label(args.update, args.label)
        print(f"Record {args.update} {'updated.' if ok else 'not found.'}")
        return EXIT_OK if ok else EXIT_ERROR

    rows = records.list_records(limit=args.limit, label_filter=args.filter)
    _print_header(f"Detection records (showing {len(rows)})")
    if not rows:
        print("  No records yet. Run `detect-image` first.")
        return EXIT_OK

    print(f"  {'ID':<5} {'SOURCE':<28} {'LABEL':<14} {'CONF':<6} {'WHEN':<20} BY")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r.record_id:<5} {r.source[:27]:<28} {r.label:<14} "
              f"{r.confidence:<6.2f} {r.created_at:<20} {r.created_by}")
    return EXIT_OK


def cmd_evaluate(args) -> int:
    """Evaluate the trained CNN on a held-out test set and print metrics."""
    from src.evaluation import evaluate_model, format_report
    from src.model import load_trained_model

    if args.data_dir:
        from train_model import load_real_dataset
        X, y = load_real_dataset(args.data_dir)
    else:
        from src.utils import generate_synthetic_dataset
        print("No --data-dir given: evaluating on a freshly generated synthetic test set.")
        X, y = generate_synthetic_dataset(n_per_class=args.samples_per_class)

    model = load_trained_model()
    result = evaluate_model(model, X, y)

    _print_header("Model evaluation")
    print(format_report(result))
    return EXIT_OK


def cmd_report(args) -> int:
    _bootstrap_database()
    _authenticate(args)

    summary = analytics.compute_summary()
    _print_header("Compliance summary")
    print(f"  Total detections   : {summary['total_detections']}")
    print(f"  With mask          : {summary['with_mask']}")
    print(f"  Without mask       : {summary['without_mask']}")
    print(f"  Compliance rate    : {summary['compliance_rate_pct']}%")
    print(f"  Average confidence : {summary['average_confidence']}")

    if summary["total_detections"] == 0:
        print("\n  Nothing to report yet. Run `detect-image` first.")
        return EXIT_OK

    path = analytics.generate_report()
    print(f"\n  Report written to: {path}")
    print(f"  CSV export in    : {config.REPORTS_DIR}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safevision",
        description="SafeVision - CV-based face mask compliance monitoring system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python main.py detect-image --path data/sample_images -u admin -p admin123",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_auth_flags(p):
        p.add_argument("-u", "--username", help="Login username (or set SAFEVISION_USER).")
        p.add_argument("-p", "--password", help="Login password (or set SAFEVISION_PASS).")

    # init
    p_init = sub.add_parser("init", help="Initialize database, default admin and sample images.")
    p_init.set_defaults(func=cmd_init)

    # register
    p_reg = sub.add_parser("register", help="Register a new user.")
    p_reg.add_argument("new_username", help="Username to create.")
    p_reg.add_argument("--new-password", help="Password (prompted if omitted).")
    p_reg.add_argument("--role", choices=auth.VALID_ROLES, default="operator")
    p_reg.set_defaults(func=cmd_register)

    # detect-image
    p_img = sub.add_parser("detect-image", help="Run detection on an image file or a folder.")
    p_img.add_argument("--path", required=True, help="Image file or directory of images.")
    p_img.add_argument("--save-annotated", action="store_true",
                       help="Write annotated copies to reports/annotated/.")
    add_auth_flags(p_img)
    p_img.set_defaults(func=cmd_detect_image)

    # detect-video
    p_vid = sub.add_parser("detect-video", help="Run detection on a video file or webcam.")
    p_vid.add_argument("--source", required=True, help="Video path, or webcam index e.g. 0.")
    p_vid.add_argument("--every-n", type=int, default=15, help="Analyze every Nth frame.")
    p_vid.add_argument("--max-frames", type=int, default=None, help="Stop after N analyzed frames.")
    p_vid.add_argument("--display", action="store_true", help="Show preview window (needs GUI).")
    p_vid.add_argument("--output", help="Write annotated mp4 to this path.")
    add_auth_flags(p_vid)
    p_vid.set_defaults(func=cmd_detect_video)

    # records
    p_rec = sub.add_parser("records", help="List, correct, or delete detection records.")
    p_rec.add_argument("--limit", type=int, default=20)
    p_rec.add_argument("--filter", choices=config.CLASS_NAMES, help="Filter by label.")
    p_rec.add_argument("--update", type=int, metavar="ID", help="Correct the label of a record.")
    p_rec.add_argument("--label", choices=config.CLASS_NAMES, help="New label for --update.")
    p_rec.add_argument("--delete", type=int, metavar="ID", help="Delete a record (admin only).")
    add_auth_flags(p_rec)
    p_rec.set_defaults(func=cmd_records)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate the trained model (accuracy, F1, confusion matrix).")
    p_eval.add_argument("--data-dir", help="Test set with class subfolders. Omit for synthetic.")
    p_eval.add_argument("--samples-per-class", type=int, default=100)
    p_eval.set_defaults(func=cmd_evaluate)

    # report
    p_rep = sub.add_parser("report", help="Generate a compliance analytics report.")
    add_auth_flags(p_rep)
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except auth.AuthError as exc:
        print(f"\n[AUTH ERROR] {exc}")
        return EXIT_AUTH
    except Exception as exc:  # noqa: BLE001
        # Imported here because src.detection pulls in OpenCV, which the
        # lightweight commands should not have to load.
        from src.detection import ModelUnavailableError
        if isinstance(exc, ModelUnavailableError):
            print(f"\n[SETUP REQUIRED] {exc}")
            return EXIT_ERROR
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unhandled error during command '%s'", args.command)
        print(f"\n[ERROR] {exc}")
        print(f"See {config.LOG_FILE} for the full traceback.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
