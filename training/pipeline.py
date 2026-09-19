"""Run the complete data and Shared BP Core training pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent


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


def main(argv=None):
    args = parse_args(argv)
    env = build_env(args.device)
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

        if not args.skip_audit:
            run_stage("审计逐回合 BP 数据", ["-m", "evaluation.audit_bp_data"], env, progress)

        shared_command = ["-m", "training.shared_bp"]
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
