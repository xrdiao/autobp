from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from sw_ui_detection.detector import SummonersWarUIDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect Summoners War UI and portrait slots from a screenshot.")
    parser.add_argument("--image", required=True, help="Path to the screenshot image.")
    parser.add_argument("--json-out", help="Optional path to save detection result json.")
    parser.add_argument("--debug-out", help="Optional path to save annotated preview image.")
    parser.add_argument(
        "--no-validate-monsters",
        action="store_true",
        help="Skip icon-to-template validation and only return structural UI detection.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    detector = SummonersWarUIDetector()
    result = detector.detect(args.image, validate_monsters=not args.no_validate_monsters)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    if args.debug_out:
        debug_path = Path(args.debug_out)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = detector.annotate(args.image, result)
        cv2.imwrite(str(debug_path), annotated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
