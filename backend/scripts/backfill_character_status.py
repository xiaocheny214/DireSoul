#!/usr/bin/env python3
"""回填角色发布状态迁移脚本。

该脚本用于修复历史数据：将所有 status=1 但实际没有真实帧的角色
更新为 status=0（草稿）。

使用方法：
    python scripts/backfill_character_status.py

注意：执行前请确保数据库连接配置正确。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, update

from windup_common.enums.character import CharacterStatus
from windup_framework.db import SessionLocal
from windup_app.server.character.model import Character


def backfill_character_status():
    """回填角色发布状态。"""
    print("开始回填角色发布状态...")

    with SessionLocal() as session:
        # 查询所有角色
        stmt = select(Character)
        characters = list(session.scalars(stmt))

        total = len(characters)
        updated = 0

        for character in characters:
            # 根据 character_data 推断正确的状态
            correct_status = CharacterStatus.from_character_data(character.character_data)

            # 如果状态不一致，更新它
            if character.status != correct_status:
                character.status = correct_status
                updated += 1
                print(f"  更新角色 ID={character.id}: {character.status} -> {correct_status}")

        # 提交更改
        session.commit()

        print(f"\n完成！共处理 {total} 个角色，更新了 {updated} 个角色的状态。")


if __name__ == "__main__":
    backfill_character_status()
