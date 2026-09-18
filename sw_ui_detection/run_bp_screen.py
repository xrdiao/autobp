from __future__ import annotations

from html import escape
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bp_core.transformer_common import PROJECT_ROOT
from sw_ui_detection.bp_screen_analyzer import BPDraftScreenAnalyzer

INPUT_DIR = PROJECT_ROOT / "sw_ui_detection" / "input"
OUTPUT_DIR = PROJECT_ROOT / "sw_ui_detection" / "output"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_input_images() -> list[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = [path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def write_index_report(rows: list[dict]) -> Path:
    index_path = OUTPUT_DIR / "bp_report_index.html"
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>BP 识别核验汇总</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:24px;}",
        "h1{margin:0 0 18px 0;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;}",
        ".card{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:14px;}",
        ".card img{width:100%;height:auto;border-radius:10px;border:1px solid #374151;background:#0b1220;margin-top:10px;}",
        ".meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}",
        ".pill{padding:5px 8px;border-radius:999px;background:#334155;font-size:12px;}",
        ".pill.ok{background:#14532d;color:#dcfce7;}",
        ".pill.bad{background:#7f1d1d;color:#fee2e2;}",
        "a{color:#93c5fd;text-decoration:none;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>BP 识别核验汇总</h1>",
        "<div class=\"grid\">",
    ]
    for row in rows:
        lines.extend(
            [
                "<div class=\"card\">",
                f"<h2 style=\"margin:0;font-size:20px;\">{escape(row['image_name'])}</h2>",
                "<div class=\"meta\">",
                f"<span class=\"pill {'ok' if row['is_bp_screen'] else 'bad'}\">BP={escape(str(row['is_bp_screen']))}</span>",
                f"<span class=\"pill\">conf={row['confidence']:.3f}</span>",
                f"<span class=\"pill\">first={escape(row['first_pick_side'])}</span>",
                f"<span class=\"pill\">current={escape(row['current_pick_side'])}</span>",
                "</div>",
                f"<div style=\"margin-top:10px;\"><a href=\"{escape(row['report_name'])}\">打开详细核验报告</a></div>",
                f"<img src=\"{escape(row['preview_name'])}\" alt=\"{escape(row['image_name'])}\">",
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body>", "</html>"])
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main() -> int:
    images = find_input_images()
    if not images:
        print(f"[INFO] put bp screenshots into: {INPUT_DIR}")
        print("[INFO] then run: python sw_ui_detection/run_bp_screen.py")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = BPDraftScreenAnalyzer()
    report_rows: list[dict] = []

    for image_path in images:
        json_path = OUTPUT_DIR / f"{image_path.stem}_bp.json"
        preview_path = OUTPUT_DIR / f"{image_path.stem}_bp_preview.png"
        report_path = OUTPUT_DIR / f"{image_path.stem}_bp_report.html"
        analysis = analyzer.save_result(
            image_path,
            output_json_path=json_path,
            output_preview_path=preview_path,
            output_report_path=report_path,
            validate_monsters=True,
        )
        report_rows.append(
            {
                "image_name": image_path.name,
                "preview_name": preview_path.name,
                "report_name": report_path.name,
                "is_bp_screen": analysis.is_bp_screen,
                "confidence": analysis.confidence,
                "first_pick_side": analysis.first_pick_side,
                "current_pick_side": analysis.current_pick_side,
            }
        )
        print(
            f"[OK] {image_path.name} "
            f"bp={analysis.is_bp_screen} conf={analysis.confidence:.3f} "
            f"first_pick={analysis.first_pick_side} current_pick={analysis.current_pick_side} "
            f"json={json_path.name} preview={preview_path.name} report={report_path.name}"
        )
    index_path = write_index_report(report_rows)
    print(f"[OK] report index: {index_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
