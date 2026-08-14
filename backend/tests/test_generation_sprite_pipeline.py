"""SpritePipeline 接入 orchestrator(离线):动作任务走新管线 → COMPLETED。

用注入的桩 SpritePipeline(不联网、不碰 onnx / 视频),验证**接线正确**:
``use_sprite_pipeline=True`` 时 run_action_task 走 SpritePipeline 分支、按动作类型构建提示词、
正确判定循环性 / 抽帧判据、把帧缩到项目 sprite 尺寸、逐帧上传并落 COMPLETED。

真实产出效果(联网、真 i2v + Color-Matting)由部署侧带 .env 跑真实端点验证,不在离线测试内。
"""
from __future__ import annotations

import io

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from windup_framework.db.base import Base
from windup_app.server.project.model import Project  # 注册 windup_project 表
from windup_app.server.orchestrator.executor import ActionTaskExecutor
from windup_app.server.orchestrator.model import (
    ActionType,
    CharacterActionInput,
    CharacterActionOutput,
    TaskStatus,
)
from windup_app.server.orchestrator.service import AiGenerationService


def _png(w: int, h: int) -> bytes:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.paste((200, 60, 60, 255), (w // 4, h // 4, w // 2, h // 2))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class _SpyPipeline:
    """记录 animate 的入参,返回固定张数的合成帧(尺寸与项目不同,以验证缩放生效)。"""

    def __init__(self) -> None:
        self.seen: dict = {}

    def animate(self, pose, action_prompt, *, n_frames, cyclic, seconds, kind, sidecar):
        self.seen = {
            "prompt": action_prompt, "n_frames": n_frames, "cyclic": cyclic,
            "seconds": seconds, "kind": kind, "sidecar": sidecar, "pose_len": len(pose),
        }
        return [_png(80, 120) for _ in range(n_frames)]

    def generate_action_frame(self, reference, action_prompt):
        # 首帧走这条(image-to-image);返回一张非项目尺寸的图,用来验证首帧**不缩放**。
        self.seen = {"i2i_prompt": action_prompt, "ref_len": len(reference)}
        return _png(100, 140)


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _make_project(session_factory, *, perspective=1, sprite=(64, 64)) -> int:
    with session_factory() as s:
        proj = Project(
            user_id=1, project_name="p", character_perspective=perspective,
            directional_movement=1, sprite_width=sprite[0], sprite_height=sprite[1],
        )
        s.add(proj)
        s.commit()
        return proj.id


def _run(session_factory, spy, action_input, project_id):
    uploaded: list[bytes] = []

    def _upload(png: bytes) -> str:
        uploaded.append(png)
        return f"https://cdn.example.com/f-{len(uploaded)}.png"

    executor = ActionTaskExecutor(
        sprite_pipeline=spy,
        upload=_upload,
        fetch_master=lambda _i: _png(64, 96),
        session_factory=session_factory,
    )
    with session_factory() as s:
        task = AiGenerationService().generate_character_action(s, user_id=1, input=action_input)
        s.commit()
        task_id = task.id
    executor.run_action_task(task_id, action_input, project_id)
    with session_factory() as s:
        done = AiGenerationService().get_task(s, project_id=project_id, task_id=task_id)
    return done, uploaded


def test_walk_runs_through_sprite_pipeline(session_factory):
    project_id = _make_project(session_factory, sprite=(64, 64))
    spy = _SpyPipeline()
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.WALK, num_frames=3, use_sprite_pipeline=True,
    )
    done, uploaded = _run(session_factory, spy, action_input, project_id)

    assert done.status is TaskStatus.COMPLETED
    assert isinstance(done.result, CharacterActionOutput)
    assert done.result.action_type == "walk"
    assert len(done.result.frames) == 3
    # 循环动作 → cyclic True;提示词非空;帧被缩到项目 sprite 尺寸(80x120 → 64x64)
    assert spy.seen["cyclic"] is True
    assert spy.seen["kind"] == "swing"
    assert spy.seen["prompt"].strip()
    assert spy.seen["sidecar"].endswith(".json")
    for png in uploaded:
        assert Image.open(io.BytesIO(png)).size == (64, 64)
    for i, frame in enumerate(done.result.frames):
        assert frame.index == i
        assert frame.image_url.startswith("https://")


def test_first_frame_uses_image_to_image_not_video(session_factory):
    """num_frames==1(动作首帧):走 image-to-image,**不碰视频**,且原样交付不缩放。"""
    project_id = _make_project(session_factory, sprite=(64, 64))
    spy = _SpyPipeline()
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.WALK, num_frames=1, use_sprite_pipeline=True,
    )
    done, uploaded = _run(session_factory, spy, action_input, project_id)

    assert done.status is TaskStatus.COMPLETED
    assert done.result.action_type == "walk"
    assert len(done.result.frames) == 1
    assert done.result.frames[0].index == 0
    assert done.result.frames[0].duration_ms is None
    # 走了 generate_action_frame(图生图),**没走** animate(视频)
    assert "i2i_prompt" in spy.seen and spy.seen["i2i_prompt"].strip()
    assert "n_frames" not in spy.seen
    # 首帧不缩放:上传的就是 i2i 原图尺寸(100x140),不是项目 sprite 尺寸(64x64)
    assert len(uploaded) == 1
    assert Image.open(io.BytesIO(uploaded[0])).size == (100, 140)


def test_attack_is_oneshot_swing(session_factory):
    project_id = _make_project(session_factory)
    spy = _SpyPipeline()
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.ATTACK, num_frames=2, use_sprite_pipeline=True,
    )
    done, _ = _run(session_factory, spy, action_input, project_id)
    assert done.status is TaskStatus.COMPLETED
    assert spy.seen["cyclic"] is False
    assert spy.seen["kind"] == "swing"


def test_jump_is_oneshot_airborne(session_factory):
    project_id = _make_project(session_factory)
    spy = _SpyPipeline()
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.JUMP, num_frames=2, use_sprite_pipeline=True,
    )
    done, _ = _run(session_factory, spy, action_input, project_id)
    assert done.status is TaskStatus.COMPLETED
    assert spy.seen["cyclic"] is False
    assert spy.seen["kind"] == "airborne"


def test_custom_without_loop_is_oneshot_and_keeps_prompt(session_factory):
    project_id = _make_project(session_factory)
    spy = _SpyPipeline()
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.CUSTOM, num_frames=2,
        custom_prompt="wave hello with the right hand", use_sprite_pipeline=True,
    )
    done, _ = _run(session_factory, spy, action_input, project_id)
    assert done.status is TaskStatus.COMPLETED
    assert spy.seen["cyclic"] is False          # loop 未给 → 兜成一次性
    assert "wave hello with the right hand" in spy.seen["prompt"]


class _BoomPipeline:
    def animate(self, *a, **k):
        raise AssertionError("flag 关闭时不应走 SpritePipeline")


class _StubGenerator:
    """现役路线的桩:按 canvas 出帧,全程离线。"""

    def generate(self, card, action, master, progress, canvas=None):
        from windup_ai_engine.ports import ActionQuality, GeneratedAction

        w, h = canvas or (64, 64)
        return GeneratedAction(
            frames=[_png(w, h) for _ in range(action.n_frames)],
            durations=[100] * action.n_frames,
            quality=ActionQuality(motion_scale=1.0, dead_frames=(), loop_seam=None),
        )


def test_default_path_untouched_when_flag_off(session_factory):
    """flag 关闭时走现役 generator,SpritePipeline 一步都不碰(注入会炸的桩确保)。"""
    project_id = _make_project(session_factory, sprite=(64, 64))
    executor = ActionTaskExecutor(
        generator=_StubGenerator(),
        sprite_pipeline=_BoomPipeline(),          # 被调用即 AssertionError
        upload=lambda _png: "https://cdn.example.com/f.png",
        fetch_master=lambda _i: _png(64, 96),
        session_factory=session_factory,
    )
    action_input = CharacterActionInput(
        character_id=1, action_type=ActionType.WALK, num_frames=2, use_sprite_pipeline=False,
    )
    with session_factory() as s:
        task = AiGenerationService().generate_character_action(s, user_id=1, input=action_input)
        s.commit()
        task_id = task.id
    executor.run_action_task(task_id, action_input, project_id)
    with session_factory() as s:
        done = AiGenerationService().get_task(s, project_id=project_id, task_id=task_id)
    assert done.status is TaskStatus.COMPLETED       # 走现役路线成功,未触发 _BoomPipeline
