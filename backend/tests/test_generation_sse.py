"""生成任务 SSE 推送的契约测试（Refs #78）。

这些断言不是随手写的，每一条都对应前端 `entities/generation/api.ts` 里会**抛错断流**
的校验：事件名必须是 ``task_update``、payload 必须带 ``task_id`` / ``task_type`` /
``status``，且 ``failed`` 必须带 ``error_message``、非 failed 必须不带。
服务端一旦违反其中任何一条，前端不是降级而是直接把流关掉。
"""

from __future__ import annotations

import dataclasses

import pytest

from windup_app.server.orchestrator.model import (
    CharacterImageOutput,
    GenerationTask,
    GenerationType,
    TaskStatus,
)
from windup_app.web.sse import stream as sse


def _task(status: TaskStatus, *, result=None, error=None) -> GenerationTask:
    return GenerationTask(
        id=7,
        user_id=1,
        project_id=3,
        task_type=GenerationType.CHARACTER_IMAGE,
        status=status,
        input_payload={},
        result=result,
        error_message=error,
    )


def _parse(chunk: str) -> list[tuple[str, str]]:
    """把 SSE 文本拆成 [(event, data)]；注释行（心跳）与 retry 指令跳过。"""
    out = []
    event = None
    for line in chunk.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            out.append((event, line[6:]))
            event = None
    return out


def test_payload_shape_matches_frontend_contract():
    """完成态：带 result、不带 error_message。"""
    task = _task(TaskStatus.COMPLETED, result=CharacterImageOutput(image_urls=["http://x/a.png"]))
    p = sse._payload(task)
    assert p["task_id"] == 7
    assert p["task_type"] == "character_image"
    assert p["status"] == "completed"
    assert p["result"] == dataclasses.asdict(task.result)
    # 前端硬约束：非 failed 带了 error_message 会被判为非法并断流
    assert p["error_message"] is None


def test_failed_task_always_carries_error_message():
    """失败态必须带 error_message —— 即使 executor 没记原因，也要兜一句文案。

    前端对 ``failed`` 但 ``error_message`` 为空的事件是直接抛错的，
    所以这里不能原样透传 None。
    """
    p = sse._payload(_task(TaskStatus.FAILED, error=None))
    assert p["status"] == "failed"
    assert isinstance(p["error_message"], str) and p["error_message"]


def test_non_failed_never_leaks_error_message():
    """跑到一半失败又被重试的任务，DB 里可能残留 error_message，不能漏给前端。"""
    p = sse._payload(_task(TaskStatus.RUNNING, error="上一次的残留"))
    assert p["status"] == "running"
    assert p["error_message"] is None


def test_sse_frame_format():
    """事件名固定 task_update，data 是单行 JSON。"""
    frame = sse._sse("task_update", {"task_id": 7, "status": "running"})
    assert frame.startswith("event: task_update\n")
    assert frame.endswith("\n\n")
    parsed = _parse(frame)
    assert parsed == [("task_update", '{"task_id": 7, "status": "running"}')]


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_terminal_task_streams_once_and_closes(client, monkeypatch, status):
    """订阅时任务已是终态：立刻推一条然后关流，不能把客户端挂在那里等。"""
    task = _task(
        status,
        result=CharacterImageOutput(image_urls=["http://x/a.png"])
        if status is TaskStatus.COMPLETED
        else None,
        error="boom" if status is TaskStatus.FAILED else None,
    )
    monkeypatch.setattr(sse, "_load", lambda task_id, project_id: task)

    with client.stream("GET", "/generation/tasks/7/stream?project_id=3") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    events = _parse(body)
    assert len(events) == 1, f"终态应只推一条就关流，实际推了 {len(events)} 条"
    name, data = events[0]
    assert name == "task_update"
    assert f'"status": "{status.value}"' in data
    # 断线重连指令要下发，前端依赖原生 EventSource 重连而不是退回轮询
    assert body.startswith("retry:")


def test_missing_task_reports_failed_instead_of_hanging(client, monkeypatch):
    """任务不存在时给一条 failed 终态，而不是让前端干等到超时。"""
    monkeypatch.setattr(sse, "_load", lambda task_id, project_id: None)

    with client.stream("GET", "/generation/tasks/999/stream?project_id=3") as resp:
        body = "".join(resp.iter_text())

    events = _parse(body)
    assert len(events) == 1
    assert '"status": "failed"' in events[0][1]
    assert '"error_message"' in events[0][1]
