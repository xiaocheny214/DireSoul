"""Fal 队列协议的视频 provider(veo / seedance 等 —— 七牛以 Fal 格式转发)。

与 :class:`.sufy.SufyVideoProvider`(kling,OpenAI 协议 ``/videos``)并列,这条走 Fal 队列:
  POST {base}/queue/fal-ai/{model}/image-to-video
  Header: Authorization: **Key** <api_key>          # 不是 Bearer
  异步:submit → status_url 轮询 → response_url → result.video.url → 下载 mp4
  body: prompt(必) + image_url(必,**公网 URL**) + duration/resolution/generate_audio

关键:Fal 侧 ``image_url`` 要**公网可取的 URL**(实测 data URI 走这条会失败),故首帧 bytes
先经注入的 ``upload`` 回调传对象存储换公网 URL —— provider 不依赖 app 层(分层门禁),
上传实现由调用方注入(bootstrap 注入 Kodo 上传)。key 由 ``AIProviderSettings`` 注入。
"""
from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from windup_framework.config.provider import AIProviderSettings, settings

from .interfaces import VideoProvider


class FalVideoProvider(VideoProvider):
    """veo/seedance i2v(Fal 队列)。首帧 bytes → (注入上传→公网URL) → mp4 bytes。"""

    def __init__(
        self,
        upload: Callable[[bytes], str],
        config: AIProviderSettings = settings,
        model: str = "veo3.1",
        duration: str = "8s",       # veo3.1 只接受特定档(实测 "5s" → 400);"8s" 已验证
        resolution: str = "720p",
        poll_interval: float = 15.0,
        max_min: int = 8,
    ) -> None:
        self._upload = upload
        self._cfg = config
        self._model = model
        self._duration = duration
        self._resolution = resolution
        self._poll = poll_interval
        self._max_min = max_min

    def _base(self) -> str:
        """Fal 端点用主机根(不带 /v1)。"""
        return self._cfg.normalized_base_url.rstrip("/").removesuffix("/v1")

    def _headers(self) -> dict:
        return {"Authorization": f"Key {self._cfg.api_key}"}

    def i2v(
        self, first_frame: bytes, prompt: str, seconds: int = 5, size: str = "1280x720"
    ) -> bytes:
        image_url = self._upload(first_frame)          # 首帧 → 公网 URL
        body = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": self._duration,
            "resolution": self._resolution,
            "generate_audio": False,
        }
        url = f"{self._base()}/queue/fal-ai/{self._model}/image-to-video"
        with httpx.Client(timeout=self._cfg.timeout, headers=self._headers()) as c:
            job = c.post(url, json=body).raise_for_status().json()
            status_url = job.get("status_url") or job.get("status")
            response_url = job.get("response_url") or job.get("response")
            if not status_url:
                raise RuntimeError(f"Fal submit 未返回 status_url: {job}")
            for _ in range(max(1, int(self._max_min * 60 // self._poll))):
                time.sleep(self._poll)
                st = c.get(status_url).raise_for_status().json()
                s = str(st.get("status", "")).upper()
                if s in ("COMPLETED", "SUCCEEDED", "OK"):
                    break
                if s in ("FAILED", "ERROR", "CANCELLED"):
                    raise RuntimeError(f"Fal i2v 失败: {st}")
            res = c.get(response_url).raise_for_status().json()
            vid = (((res.get("result") or res).get("video") or {}).get("url")
                   or (res.get("video") or {}).get("url"))
            if not vid:
                raise RuntimeError(f"Fal 未取得 video url: {res}")
            return c.get(vid).raise_for_status().content
