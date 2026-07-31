#!/usr/bin/env python3
"""种子:把 3D 角色 xed(bare 变体,5 动作)种进后端,供 playtest 展示。

素材:~/jz_code/windup-pipeline/rigmvp/frames3d/bare/<action>/f*.png(透明,1107x924)。
root_motion 用 meta.json 的**真 speed**(walk 1.14、run 3.02 身高/秒)算累积 dx;
jump 竖直已烘进帧、idle/attack 无位移 → dy/dx 为 0。durations/fps 用 meta 的 dur/frames。
从 backend/ 跑:.venv/bin/python <this>。
"""
from __future__ import annotations
import io, json, uuid
from pathlib import Path

import numpy as np
from PIL import Image

from windup_app.server.character.model import Character
from windup_app.server.media.model import MediaUploadInput
from windup_app.server.media.service import ObjectStorageMediaService
from windup_app.server.project.model import Project
from windup_common.enums.media import MediaCategory
from windup_framework.db import SessionLocal

RIG = Path.home() / "jz_code" / "windup-pipeline" / "rigmvp"
BARE = RIG / "frames3d" / "bare"
META = json.loads((RIG / "frames3d" / "meta.json").read_text())
SPRITE = 512
ACTIONS = ["idle", "walk", "run", "jump", "attack"]
TYPE_MAP = {"idle": "idle", "walk": "walk", "run": "custom", "jump": "jump", "attack": "attack"}
NAME_MAP = {"idle": "待机", "walk": "行走", "run": "奔跑", "jump": "跳跃", "attack": "攻击"}


def _fit(png: bytes) -> tuple[bytes, int]:
    """帧等比缩进 SPRITE×SPRITE 透明画布;返回 (bytes, 角色bbox高px)。"""
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    im.thumbnail((SPRITE, SPRITE), Image.LANCZOS)
    canvas = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((SPRITE - im.width) // 2, (SPRITE - im.height) // 2))
    a = np.asarray(canvas)[:, :, 3] > 128
    ys, _ = np.where(a)
    h = int(ys.max() - ys.min()) if len(ys) else SPRITE
    buf = io.BytesIO(); canvas.save(buf, "PNG")
    return buf.getvalue(), h


def _frames(action: str) -> list[Path]:
    return sorted((BARE / action).glob("f*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))


def main() -> None:
    media = ObjectStorageMediaService()
    outfit_id = f"xed-bare-{uuid.uuid4().hex[:8]}"
    actions_data = []

    for action in ACTIONS:
        m = META["actions"][action]
        paths = _frames(action)
        n = len(paths)
        dur_s, speed = m["dur"], m.get("speed", 0.0)
        duration_ms = max(30, round(dur_s * 1000 / n))
        fps = max(1, round(n / dur_s))
        frames = []
        char_h = SPRITE
        for i, p in enumerate(paths):
            fitted, h = _fit(p.read_bytes())
            char_h = h if i == 0 else char_h
            url = media.upload(
                fitted,
                MediaUploadInput(filename=f"xed_{action}_{i:02d}.png", content_type="image/png",
                                 size=len(fitted), category=MediaCategory.ACTION_FRAME),
            ).url
            # root_motion:walk/run 水平累积(真 speed×dur×身高);jump 竖直已烘进帧;idle/attack 无。
            if action in ("walk", "run") and n > 1:
                total_dx = speed * dur_s * char_h
                dx = round(i / (n - 1) * total_dx)
            else:
                dx = 0
            frames.append({"index": i, "image_url": url, "duration_ms": duration_ms,
                           "root_motion": {"dx": dx, "dy": 0}})
            print(f"  {action} {i + 1}/{n} -> {url}", flush=True)
        actions_data.append({
            "id": f"{outfit_id}-{action}", "type": TYPE_MAP[action], "name": NAME_MAP[action],
            "loop": bool(m.get("loop", False)), "fps": fps, "frame_count": n, "frames": frames,
        })

    preview = actions_data[0]["frames"][0]["image_url"]
    character_data = {"version": 1, "outfits": [{
        "id": outfit_id, "name": "xed · 3D 粘土", "description": "3D 渲染的 2D 序列帧(bare)",
        "preview_url": preview, "actions": actions_data,
    }]}

    with SessionLocal() as session:
        project = Project(user_id=1, name="xed 3D 演示", character_perspective=1,
                          directional_movement=1, sprite_width=SPRITE, sprite_height=SPRITE,
                          game_style="3D-rendered clay, side view")
        session.add(project); session.flush()
        project_id = project.id
        character = Character(user_id=1, project_id=project_id,
                              description="xed —— 3D 建模渲染的 2D 角色,带真实 root motion",
                              reference_image_url=preview, character_data=character_data, status=1)
        session.add(character); session.flush()
        character_id = character.id
        session.commit()

    print(f"\nPROJECT_ID={project_id} CHARACTER_ID={character_id} OUTFIT_ID={outfit_id}", flush=True)
    print(f"playtest: /playtest/{character_id}/{outfit_id}", flush=True)


if __name__ == "__main__":
    main()
