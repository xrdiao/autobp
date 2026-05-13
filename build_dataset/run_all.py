import argparse
import asyncio

from batch_extract_match_bp import batch_extract_match_bp
from build_monster_catalog import build_catalog
from download_players_by_rank import download_players_by_rank
from expand_from_matches_raw import expand_from_matches_raw
from generate_alias_and_images import generate_alias_and_images
from normalize_training_samples import normalize_training_samples
from rebuild_monster_images import rebuild_monster_images
from sync_images import sync_images


def parse_args():
    parser = argparse.ArgumentParser(description="Run the BP dataset build pipeline.")
    parser.add_argument("--download", action="store_true", help="Download player pages before extraction.")
    parser.add_argument("--skip-images", action="store_true", help="Skip image sync and rebuild steps.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild expanded samples instead of incremental append.")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalized me-only sample export.")
    parser.add_argument("--use-existing-version-csv", action="store_true", help="Reuse the current version CSV when downloading.")
    parser.add_argument("--detail-concurrency", type=int, default=1, help="Detail-page concurrency for polite crawling.")
    parser.add_argument("--max-detail-fetches", type=int, default=180, help="Max number of detail pages to fetch in one run.")
    parser.add_argument("--detail-min-interval-seconds", type=float, default=3.0, help="Min delay between detail requests.")
    parser.add_argument("--detail-max-interval-seconds", type=float, default=5.0, help="Max delay between detail requests.")
    parser.add_argument("--extract-max-workers", type=int, default=None, help="Worker count for html extraction.")
    parser.add_argument("--expand-max-workers", type=int, default=None, help="Worker count for match expansion.")
    parser.add_argument("--expand-fetch-batch-size", type=int, default=1000, help="SQLite fetch batch size when expanding raw matches.")
    parser.add_argument("--export-pt", action="store_true", help="Also export legacy .pt datasets.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download:
        asyncio.run(
            download_players_by_rank(
                use_existing_csv=args.use_existing_version_csv,
                detail_concurrency=args.detail_concurrency,
                max_detail_fetches=args.max_detail_fetches,
                detail_min_interval_seconds=args.detail_min_interval_seconds,
                detail_max_interval_seconds=args.detail_max_interval_seconds,
            )
        )

    batch_extract_match_bp(
        incremental=not args.force_rebuild,
        max_workers=args.extract_max_workers,
        export_dataset_pt=args.export_pt,
    )
    expand_from_matches_raw(
        incremental=not args.force_rebuild,
        fetch_batch_size=args.expand_fetch_batch_size,
        max_workers=args.expand_max_workers,
        export_pt=args.export_pt,
    )
    build_catalog("dataset", "json//monster_catalog.json")
    generate_alias_and_images()
    if not args.skip_normalize:
        normalize_training_samples()
    if not args.skip_images:
        sync_images()
        rebuild_monster_images()


if __name__ == "__main__":
    main()
