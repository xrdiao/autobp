from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync current BP data, audit samples, and train all recommendation models."
    )
    parser.add_argument(
        "--sync-web-data",
        action="store_true",
        help="Sync Lucksack, SWKit, SWRT statistics, and SWARFARM mechanisms first.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the chronological BP sample audit.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train relation v2 from scratch and rebuild its split manifest.",
    )
    parser.add_argument("--skip-relation", action="store_true", help="Skip relation v2 training.")
    parser.add_argument("--skip-prefix", action="store_true", help="Skip prefix value training.")
    parser.add_argument("--skip-draft", action="store_true", help="Skip full-draft value training.")
    parser.add_argument(
        "--relation-epochs",
        type=int,
        default=None,
        help="Override relation epochs. Defaults to the trainer's fine-tune/from-scratch setting.",
    )
    parser.add_argument(
        "--prefix-epochs",
        type=int,
        default=None,
        help="Override prefix value epochs. Defaults to the trainer setting.",
    )
    parser.add_argument(
        "--relation-workers",
        type=int,
        default=0,
        help="Data-loader workers for relation training; 0 is the stable default.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device passed to relation/prefix models, for example cuda or cpu.",
    )
    parser.add_argument(
        "--combo-limit",
        type=int,
        default=2000,
        help="Maximum duo/trio records fetched during the Lucksack sync.",
    )
    parser.add_argument(
        "--swrt-monster-limit",
        type=int,
        default=400,
        help="Maximum SWRT monsters for which matchup detail is fetched.",
    )
    parser.add_argument(
        "--swrt-workers",
        type=int,
        default=4,
        help="Concurrent workers used for public SWRT matchup detail requests.",
    )
    return parser.parse_args()


def run_stage(label, command, env=None):
    print(f"\n=== {label} ===", flush=True)
    print("$", " ".join(command), flush=True)
    subprocess.run(
        [sys.executable, *command],
        cwd=ROOT_DIR,
        check=True,
        env=env,
    )


def build_env(device):
    if not device:
        return None
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def main():
    args = parse_args()
    env = build_env(args.device)

    if args.sync_web_data:
        run_stage(
            "同步 SWKit 别称和 RTA 评级",
            ["build_dataset/sync_swkit_data.py"],
            env=env,
        )
        run_stage(
            "同步 Lucksack 赛季、段位、BP 统计和魔灵目录",
            [
                "build_dataset/sync_lucksack_data.py",
                "--combo-limit",
                str(args.combo_limit),
            ],
            env=env,
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
            env=env,
        )
        run_stage(
            "同步 SWARFARM 技能机制",
            ["build_dataset/sync_swarfarm_mechanisms.py"],
            env=env,
        )

    if not args.skip_audit:
        run_stage(
            "审计逐回合 BP 数据",
            ["audit_bp_prefix_data.py"],
            env=env,
        )

    if not args.skip_relation:
        relation_command = [
            "train_bp_relation_net.py",
            "--rebuild-preprocessed-cache",
            "--num-workers",
            str(args.relation_workers),
        ]
        if args.from_scratch:
            relation_command.extend(["--from-scratch", "--rebuild-split"])
        else:
            relation_command.append("--fine-tune")
        if args.relation_epochs is not None:
            relation_command.extend(["--epochs", str(args.relation_epochs)])
        if args.device:
            relation_command.extend(["--device", args.device])
        run_stage("训练 relation v2", relation_command, env=env)

    if not args.skip_prefix:
        prefix_command = ["train_prefix_value_net.py"]
        if args.prefix_epochs is not None:
            prefix_command.extend(["--epochs", str(args.prefix_epochs)])
        if args.device:
            prefix_command.extend(["--device", args.device])
        run_stage("训练 prefix value", prefix_command, env=env)

    if not args.skip_draft:
        run_stage("训练完整阵容 value / Ban 模型", ["train_draft_value_net.py"], env=env)

    print("\n完成：网页统计、机制数据和模型已按当前数据快照更新。", flush=True)


if __name__ == "__main__":
    main()
