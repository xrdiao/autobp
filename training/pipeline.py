"""Run the complete data and Shared BP Core training pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

from bp_core.data_common import count_record_rows

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bp_samples.sqlite"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="同步公开数据、审计回放并训练唯一的 Shared BP Core。"
    )
    parser.add_argument(
        "--sync-web-data",
        action="store_true",
        help="先同步 Lucksack、SWKit、SWRT 和 SWARFARM 数据。",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="跳过逐回合 BP 数据审计。",
    )
    parser.add_argument(
        "--shared-epochs",
        type=int,
        default=None,
        help="覆盖 Shared BP Core 训练轮数。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="训练设备，例如 cuda 或 cpu。",
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="事件级 BP 样本 SQLite 路径。",
    )
    parser.add_argument("--combo-limit", type=int, default=2000)
    parser.add_argument("--swrt-monster-limit", type=int, default=400)
    parser.add_argument("--swrt-workers", type=int, default=4)
    return parser.parse_args(argv)


def run_stage(label, command, env, progress):
    started = time.perf_counter()
    print(f"\n=== {label} ===", flush=True)
    print("$", " ".join(command), flush=True)
    subprocess.run([sys.executable, *command], cwd=ROOT_DIR, check=True, env=env)
    elapsed = time.perf_counter() - started
    print(f"[DONE] {label} ({elapsed:.1f}s)", flush=True)
    progress.update(1)
    progress.set_postfix_str(f"{label} {elapsed:.1f}s")


def build_env(device):
    if not device:
        return None
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def require_samples(data_path):
    """Fail before audit/training when the dataset stage produced no samples."""

    path = resolve_data_path(data_path)
    sample_count = count_record_rows(path, "bp_samples")
    if sample_count <= 0:
        raise RuntimeError(
            f"没有可训练的 BP 样本: {path}. "
            "请先成功完成 `python build_dataset/run_all.py`；"
            "如果排行榜下载中断，请修复网络或验证页后重新运行，并使用"
            " `--dataset-version <已有版本> --use-existing-version-csv` 继续已有版本。"
        )
    print(f"[INFO] BP samples: {sample_count} ({path})", flush=True)
    return sample_count


def resolve_data_path(data_path):
    path = Path(data_path)
    return path if path.is_absolute() else ROOT_DIR / path


def main(argv=None):
    args = parse_args(argv)
    env = build_env(args.device)
    data_path = resolve_data_path(args.data_path)
    sync_count = 4 if args.sync_web_data else 0
    audit_count = 0 if args.skip_audit else 1
    total_stages = sync_count + audit_count + 1
    progress = tqdm(total=total_stages, desc="训练流水线", unit="stage", dynamic_ncols=True)
    try:
        if args.sync_web_data:
            run_stage("同步 SWKit 别称和 RTA 评级", ["build_dataset/sync_swkit_data.py"], env, progress)
            run_stage(
                "同步 Lucksack 赛季、段位、BP 统计和魔灵目录",
                ["build_dataset/sync_lucksack_data.py", "--combo-limit", str(args.combo_limit)],
                env,
                progress,
            )
            run_stage(
                "同步 SWRT 怪物统计和对位数据",
                [
                    "build_dataset/sync_swrt_data.py",
                    "--monster-limit",
                    str(args.swrt_monster_limit),
                    "--max-workers",
                    str(args.swrt_workers),
                ],
                env,
                progress,
            )
            run_stage("同步 SWARFARM 技能机制", ["build_dataset/sync_swarfarm_mechanisms.py"], env, progress)

        require_samples(data_path)

        if not args.skip_audit:
            run_stage(
                "审计逐回合 BP 数据",
                ["-m", "evaluation.audit_bp_data", "--data-path", str(data_path)],
                env,
                progress,
            )

        shared_command = ["-m", "training.shared_bp", "--data-path", str(data_path)]
        if args.shared_epochs is not None:
            shared_command.extend(["--epochs", str(args.shared_epochs)])
        if args.device:
            shared_command.extend(["--device", args.device])
        run_stage("训练 Shared BP Core", shared_command, env, progress)
    finally:
        progress.close()
    print("\n完成：公开数据、回放审计和 Shared BP Core 已更新。", flush=True)


if __name__ == "__main__":
    main()
