import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_dataset.batch_extract_match_bp import batch_extract_match_bp
from build_dataset.build_monster_catalog import build_catalog
from build_dataset.download_players_by_rank import download_players_by_rank
from build_dataset.expand_from_matches_raw import expand_from_matches_raw
from build_dataset.generate_alias_and_images import generate_alias_and_images
from build_dataset.rebuild_monster_images import rebuild_monster_images
from build_dataset.sync_images import sync_images
from build_dataset.sync_lucksack_data import sync_lucksack_data
from build_dataset.sync_swarfarm_mechanisms import sync_swarfarm_mechanisms
from build_dataset.sync_swrt_data import sync_swrt_data
from build_dataset.sync_swkit_data import sync_swkit_data
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run the BP dataset build pipeline.")
    parser.add_argument("--download", action="store_true", help="Download player pages before extraction.")
    parser.add_argument("--skip-images", action="store_true", help="Skip image sync and rebuild steps.")
    parser.add_argument("--skip-lucksack", action="store_true", help="Skip syncing the live Lucksack catalog and statistics.")
    parser.add_argument("--lucksack-group-concurrency", type=int, default=2, help="Concurrent Lucksack rank-group sync workers.")
    parser.add_argument("--skip-mechanisms", action="store_true", help="Skip syncing SWARFARM skill mechanism data.")
    parser.add_argument("--skip-swkit", action="store_true", help="Skip syncing SWKit aliases and RTA tiers.")
    parser.add_argument("--skip-swrt", action="store_true", help="Skip syncing public SWRT RTA statistics.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild expanded samples instead of incremental append.")
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Use a fixed dataset version for download/resume, for example v3.0.",
    )
    parser.add_argument(
        "--use-existing-version-csv",
        action="store_true",
        help="Reuse dataset/<version>/ranking_list.csv when resuming a download.",
    )
    parser.add_argument("--detail-concurrency", type=int, default=2, help="Detail-page concurrency for polite crawling.")
    parser.add_argument("--max-detail-fetches", type=int, default=180, help="Max number of detail pages to fetch in one run.")
    parser.add_argument("--detail-min-interval-seconds", type=float, default=1.5, help="Min delay between detail requests.")
    parser.add_argument("--detail-max-interval-seconds", type=float, default=2.5, help="Max delay between detail requests.")
    parser.add_argument("--extract-max-workers", type=int, default=None, help="Worker count for html extraction.")
    parser.add_argument("--expand-max-workers", type=int, default=None, help="Worker count for match expansion.")
    parser.add_argument("--expand-fetch-batch-size", type=int, default=1000, help="SQLite fetch batch size when expanding raw matches.")
    return parser.parse_args()


def main():
    args = parse_args()
    stages = []
    if not args.skip_swkit:
        stages.append(("同步 SWKit 别称和 RTA 评级", lambda: asyncio.run(sync_swkit_data())))
    if not args.skip_lucksack:
        stages.append(
            (
                "同步 Lucksack 赛季、段位、BP 统计和魔灵目录",
                lambda: asyncio.run(
                    sync_lucksack_data(group_concurrency=args.lucksack_group_concurrency)
                ),
            )
        )
    if not args.skip_swrt:
        stages.append(("同步 SWRT 怪物统计和对位数据", sync_swrt_data))
    if not args.skip_mechanisms:
        stages.append(("同步 SWARFARM 技能机制", sync_swarfarm_mechanisms))
    if args.download:
        stages.append(
            (
                "下载排行榜玩家详情",
                lambda: asyncio.run(
                    download_players_by_rank(
                        dataset_version=args.dataset_version or None,
                        use_existing_csv=args.use_existing_version_csv,
                        detail_concurrency=args.detail_concurrency,
                        max_detail_fetches=args.max_detail_fetches,
                        detail_min_interval_seconds=args.detail_min_interval_seconds,
                        detail_max_interval_seconds=args.detail_max_interval_seconds,
                    )
                ),
            )
        )
    stages.extend(
        [
            (
                "提取比赛 BP 数据",
                lambda: batch_extract_match_bp(
                    incremental=not args.force_rebuild,
                    max_workers=args.extract_max_workers,
                ),
            ),
            (
                "展开逐回合训练样本",
                lambda: expand_from_matches_raw(
                    incremental=not args.force_rebuild,
                    fetch_batch_size=args.expand_fetch_batch_size,
                    max_workers=args.expand_max_workers,
                ),
            ),
            (
                "构建魔灵目录",
                lambda: build_catalog(
                    str(PROJECT_ROOT / "dataset"),
                    str(PROJECT_ROOT / "json" / "monster_catalog.json"),
                ),
            ),
            ("生成别称和图片映射", generate_alias_and_images),
        ]
    )
    if not args.skip_images:
        stages.extend(
            [
                ("同步魔灵图片", sync_images),
                ("重建魔灵图片映射", rebuild_monster_images),
            ]
        )

    total_stages = len(stages)
    print(f"[INFO] pipeline stages: {total_stages}", flush=True)
    with tqdm(total=total_stages, desc="数据流水线", unit="stage", dynamic_ncols=True) as progress:
        for index, (label, stage) in enumerate(stages, start=1):
            started = time.perf_counter()
            print(f"\n[{index}/{total_stages}] {label}", flush=True)
            try:
                stage()
            except Exception as exc:
                progress.set_postfix_str(f"{label} 失败")
                print(
                    f"[FAILED] {label}: {exc}\n"
                    "后续阶段已停止。已完成阶段的输出会保留。",
                    flush=True,
                )
                if label == "下载排行榜玩家详情":
                    if args.dataset_version:
                        resume_hint = (
                            f"--dataset-version {args.dataset_version} "
                            "--use-existing-version-csv"
                        )
                    else:
                        resume_hint = (
                            "设置 BP_DATASET_VERSION=<已有版本>，"
                            "并加 --use-existing-version-csv"
                        )
                    print(
                        "[INFO] 修复网络或验证页后可重新运行下载；"
                        f"如需复用已有排行榜文件，请{resume_hint}。",
                        flush=True,
                    )
                raise
            elapsed = time.perf_counter() - started
            progress.update(1)
            progress.set_postfix_str(f"{label} {elapsed:.1f}s")
            print(f"[DONE] {label} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
