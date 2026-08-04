"""生成任务的 SSE 推送端点（替代前端轮询，Refs #78）。

契约以**实际消费方**为准 —— 前端 `entities/generation/api.ts` 的适配器订阅的是
单一事件名 ``task_update``：

    event: task_update
    data: {"task_id":1,"task_type":"character_image","status":"running",
           "result":null,"error_message":null}

前端会逐字段校验，其中两条是硬约束（违反直接抛错断流）：
  * ``status == "failed"`` **必须**带 ``error_message``；
  * 其余状态**必须不带** ``error_message``。
收到 ``completed`` / ``failed`` 前端自行关流。

> **契约冲突备忘**：`docs/agent-sse-api-design.md` 写的是四个事件名
> （status / progress / completed / failed），与此处的单事件 ``task_update`` 不一致。
> 前端已按 ``task_update`` 落地，故服务端先对齐前端；两份文档需要收敛成一份。

实现取「轮询数据库 + 状态变化才推」而不是进程内事件总线，原因：
executor 走 BackgroundTasks 在同进程跑，事件总线看似更快，但一旦 uvicorn 起多 worker
就会出现「订阅在 A 进程、任务在 B 进程」的静默失效。轮询 DB 无此问题，代价只是最多
一个轮询间隔的延迟，对分钟级的生成任务可以忽略。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from windup_framework.db.session import SessionLocal

from windup_app.server.orchestrator.model import TaskStatus
from windup_app.server.orchestrator.service import service as generation_service

logger = logging.getLogger("windup.generation.sse")

router = APIRouter(prefix="/generation", tags=["generation"])

POLL_SECONDS = 1.5          # 状态轮询间隔
HEARTBEAT_SECONDS = 15.0    # 无变化时发注释行保活，防中间层掐断空闲连接
MAX_STREAM_SECONDS = 1800   # 单条流最长存活；超时后关闭，由前端自动重连续订


def _payload(task) -> dict:
    """组装 task_update 的 data。字段名与前端校验逐条对齐。"""
    result = dataclasses.asdict(task.result) if task.result is not None else None
    # 前端硬约束：非 failed 不得带 error_message
    error = task.error_message if task.status == TaskStatus.FAILED else None
    if task.status == TaskStatus.FAILED and not error:
        error = "任务失败但未记录原因"       # 宁可给个兜底文案，也不能让前端因缺字段抛错
    return {
        "task_id": task.id,
        "task_type": task.task_type.value,
        "status": task.status.value,
        "result": result,
        "error_message": error,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _load(task_id: int, project_id: int):
    """同步读 DB —— 由调用方丢进线程池，别阻塞事件循环。"""
    session = SessionLocal()
    try:
        return generation_service.get_task(session, project_id, task_id)
    finally:
        session.close()


@router.get("/tasks/{task_id}/stream")
async def stream_task(task_id: int, request: Request, project_id: int = Query(..., gt=0)):
    """订阅一个生成任务的状态变化。任务已处于终态时立即推一条并关闭。"""

    async def events():
        # 告诉浏览器断线后 3 秒重连（前端依赖原生 EventSource 重连，不退回轮询）
        yield "retry: 3000\n\n"

        last: dict | None = None
        waited = 0.0
        since_beat = 0.0
        while True:
            if await request.is_disconnected():
                logger.info("客户端断开，停止推送 task=%s", task_id)
                return

            task = await run_in_threadpool(_load, task_id, project_id)
            if task is None:
                # 任务不存在：给一条 failed 让前端有明确终态，而不是干等到超时
                yield _sse("task_update", {
                    "task_id": task_id, "task_type": "character_image",
                    "status": "failed", "result": None,
                    "error_message": f"任务 {task_id} 不存在或不属于项目 {project_id}",
                })
                return

            payload = _payload(task)
            if payload != last:                      # 只在变化时推，避免刷屏
                yield _sse("task_update", payload)
                last = payload
                since_beat = 0.0

            if task.is_terminal:
                return

            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
            since_beat += POLL_SECONDS
            if since_beat >= HEARTBEAT_SECONDS:
                yield ": heartbeat\n\n"
                since_beat = 0.0
            if waited >= MAX_STREAM_SECONDS:
                logger.info("流存活超过 %ss，关闭等待前端重连 task=%s", MAX_STREAM_SECONDS, task_id)
                return

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关掉 nginx 缓冲，否则事件会被攒着一起发
        },
    )
