"""种子脚本:插入一个 Demo Project + Character(五动作骑士),供 playtest 路由展示。

用法(需在 backend/ 目录下跑,以便 .env 相对路径解析正确):

    /Users/johnnyzhang/jz_code/DireSoul/backend/.venv/bin/python seed_demo_character.py

跑完打印 `PROJECT_ID=.. CHARACTER_ID=.. OUTFIT_ID=..`,
对应 playtest 路由 `/playtest/<CHARACTER_ID>/<OUTFIT_ID>`。
"""

from __future__ import annotations

import math
import re
import uuid
from pathlib import Path

from windup_app.server.character.model import Character
from windup_app.server.media.model import MediaUploadInput
from windup_app.server.media.service import ObjectStorageMediaService
from windup_app.server.project.model import Project
from windup_common.enums.media import MediaCategory
from windup_framework.db import SessionLocal

# ── 素材来源 ──────────────────────────────────────────────────────────────
FRAMES_ROOT = Path(
    "/private/tmp/claude-501/-Users-johnnyzhang/"
    "91c539d9-70bc-412c-afe0-0686184ad21d/scratchpad/real_run_out/turbo_parallel"
)

SPRITE_W = 256
SPRITE_H = 256

# 动作 → (type 字段, 中文显示名, 是否循环)
ACTION_META: dict[str, tuple[str, str, bool]] = {
    "idle": ("idle", "待机", True),
    "walk": ("walk", "行走", True),
    "run": ("custom", "奔跑", True),
    "attack": ("attack", "攻击", False),
    "jump": ("jump", "跳跃", False),
}

# 各动作基准单帧时长(ms)—— 与 windup_ai_engine.postprocess.rootmotion.DEFAULT_FPS_MS 对齐
DEFAULT_FPS_MS: dict[str, int] = {
    "idle": 450,
    "walk": 125,
    "run": 90,
    "jump": 110,
    "attack": 90,
}

ACTION_ORDER = ["idle", "walk", "run", "attack", "jump"]

_NUM_RE = re.compile(r"(\d+)")


def _frame_sort_key(path: Path) -> int:
    m = _NUM_RE.search(path.stem)
    return int(m.group(1)) if m else 0


def _list_frame_pngs(action: str) -> list[Path]:
    """列出某动作目录下按序号排序的帧 PNG,排除 .gif / .mp4 等非帧文件。"""
    action_dir = FRAMES_ROOT / action
    pngs = sorted(
        (p for p in action_dir.glob(f"{action}_*.png") if p.suffix.lower() == ".png"),
        key=_frame_sort_key,
    )
    return pngs


def _root_motion_for(action: str, i: int, n: int) -> dict:
    """按动作类型合成 root motion(累积绝对位移,y 向上为正)。

    帧序列本身已原地对齐(align-normalized),故位移为纯合成:
    - idle / attack: 原地动作,恒为 (0, 0)
    - walk: 匀速前进,STEP = sprite_w * 0.06
    - run: 匀速前进,步幅是 walk 的 1.8 倍
    - jump: 抛物线式高度变化(半个正弦波),峰值在动作中点
    """
    if action in ("idle", "attack"):
        return {"dx": 0, "dy": 0}
    if action == "walk":
        step = SPRITE_W * 0.06
        return {"dx": round(i * step), "dy": 0}
    if action == "run":
        step = SPRITE_W * 0.06 * 1.8
        return {"dx": round(i * step), "dy": 0}
    if action == "jump":
        denom = max(n - 1, 1)
        dy = math.sin(math.pi * i / denom) * SPRITE_H * 0.5
        return {"dx": 0, "dy": round(dy)}
    return {"dx": 0, "dy": 0}


def main() -> None:
    media = ObjectStorageMediaService()

    # ── 1. 逐动作上传帧,组 CharacterAction ──────────────────────────────
    actions_payload = []
    reference_image_url: str | None = None

    for action in ACTION_ORDER:
        frame_paths = _list_frame_pngs(action)
        if not frame_paths:
            raise RuntimeError(f"动作 {action} 下没找到帧 PNG: {FRAMES_ROOT / action}")

        action_type, action_name, loop = ACTION_META[action]
        duration_ms = DEFAULT_FPS_MS[action]
        fps = round(1000 / duration_ms)
        n = len(frame_paths)

        frames_payload = []
        for i, frame_path in enumerate(frame_paths):
            data = frame_path.read_bytes()
            result = media.upload(
                data,
                MediaUploadInput(
                    filename=frame_path.name,
                    content_type="image/png",
                    size=len(data),
                    category=MediaCategory.ACTION_FRAME,
                ),
            )
            if reference_image_url is None:
                reference_image_url = result.url
            frames_payload.append(
                {
                    "index": i,
                    "image_url": result.url,
                    "duration_ms": duration_ms,
                    "root_motion": _root_motion_for(action, i, n),
                }
            )
            print(f"  上传 {action} 帧 {i + 1}/{n}: {frame_path.name} -> {result.url}")

        actions_payload.append(
            {
                "id": f"action-{action}-{uuid.uuid4().hex[:8]}",
                "type": action_type,
                "name": action_name,
                "loop": loop,
                "fps": fps,
                "frame_count": n,
                "frames": frames_payload,
            }
        )

    outfit_id = f"outfit-{uuid.uuid4().hex[:8]}"
    character_data = {
        "version": 1,
        "outfits": [
            {
                "id": outfit_id,
                "name": "默认造型",
                "description": None,
                "preview_url": reference_image_url,
                "actions": actions_payload,
            }
        ],
    }

    # ── 2. 落库:Project + Character ─────────────────────────────────────
    with SessionLocal() as session:
        project = Project(
            user_id=1,
            project_name=f"Windup演示骑士-{uuid.uuid4().hex[:6]}",
            character_perspective=1,  # 1=横版侧视
            directional_movement=1,  # 1=单向
            sprite_width=SPRITE_W,
            sprite_height=SPRITE_H,
            game_style="dark fantasy illustration",
        )
        session.add(project)
        session.flush()  # 取回 project.id

        character = Character(
            project_id=project.id,
            description="五动作骑士(idle/walk/run/attack/jump) demo",
            reference_image_url=reference_image_url,
            character_data=character_data,
            status=1,  # active
        )
        session.add(character)
        session.flush()  # 取回 character.id

        session.commit()

        project_id = project.id
        character_id = character.id

    print(
        f"PROJECT_ID={project_id} CHARACTER_ID={character_id} OUTFIT_ID={outfit_id}"
    )


if __name__ == "__main__":
    main()
