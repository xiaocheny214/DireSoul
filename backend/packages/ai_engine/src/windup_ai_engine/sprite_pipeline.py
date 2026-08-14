"""精灵动画视频管线 —— 端到端编排(参考 godogen asset-gen 的精灵动画线)。

    参考帧 ─► 动作帧 ─► 视频 ─► 抽帧 ─► 循环检测 ─► 批量抠图 ─► PNG 序列
     (1)       (2)      (3)     (4)        (5)          (6)

**本模块是「编排」,不是「算法」。** 六步里每一步的底层能力仓里都已具备,本管线只是把它们
按顺序串起来、补上参考帧 / 动作帧的生成与背景色策略,并让视频步骤可 resume。刻意复用而不
重造:

  - 参考帧 / 动作帧(1、2):``ImageProvider``(文生图 / 图生图,见 framework.providers)
  - 视频(3)              :``VideoProvider.i2v``(kling i2v;超时可 resume,见 sufy.VideoJobTimeout)
  - 抽帧(4)              :``slicing.extract_all_frames_bytes``
  - 循环检测(5)          :``slicing.pick_cycle`` / ``pick_oneshot``(比参考手册的朴素余弦窗更强:
                            去平移、谐波复选、prominence 判据,见 slicing.loop)
  - 批量抠图(6)          :``MatteProvider``(纯色背景用 ``ColorMatteProvider``,边缘更干净)

**分层**:本模块在 ai_engine 层,可依赖 framework.providers(契约)与 common;不碰存储 / DB /
业务域(那些在 app.server)。产出一律是**帧 bytes**,落盘只发生在 CLI 边界 —— 与
``ai_engine.ports`` 的「引擎只出字节」约定一致,server 侧拿字节自行上传对象存储。

**异常约定**:沿用 ai_engine 的做法 —— 直接抛类型化异常(``ValueError`` / provider 的
``VideoJobTimeout`` 等),**不**在引擎层包 ``Response[T]``;``Response`` 只在 CLI 输出边界用。

CLI(四个子命令,各自独立入口)::

    python -m windup_ai_engine.sprite_pipeline reference --name Knight \\
        --description "silver armor, blue cape, longsword" --scene forest -o ref.png
    python -m windup_ai_engine.sprite_pipeline action --reference ref.png \\
        --prompt "running forward" -o pose.png
    python -m windup_ai_engine.sprite_pipeline animate --pose pose.png \\
        --prompt "running forward" --frames 8 --cyclic --sidecar run.json -o frames/
    python -m windup_ai_engine.sprite_pipeline animate --resume --sidecar run.json -o frames/
    python -m windup_ai_engine.sprite_pipeline run --spec knight.json -o out/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from windup_ai_engine._imgio import from_png as _img
from windup_ai_engine._imgio import to_png as _png
from windup_ai_engine.ports import ProgressPort
from windup_ai_engine.slicing import extract_all_frames_bytes, pick_cycle, pick_oneshot

if TYPE_CHECKING:
    from windup_framework.providers import ImageProvider, MatteProvider, VideoProvider

__all__ = [
    "pick_bg_color",
    "build_reference_prompt",
    "SpriteActionSpec",
    "SpriteActionResult",
    "SpriteCharacterResult",
    "SpritePipeline",
    "build_default_pipeline",
    "mirror_frames",
]

# ── 成本参考(¢,来自 asset-workflow 手册 7.4;仅用于**预估**,不代表真实计费)──────
_COST_REFERENCE_CENTS = 7      # 人物模板(gemini 1K)
_COST_ACTION_CENTS = 7         # 动作帧(gemini 1K,image-to-image)
_COST_VIDEO_PER_SEC_CENTS = 5  # 视频 5¢/秒

# ── 背景色策略(手册 2.1)──────────────────────────────────────────────────────
#
# 背景色 = 与主体区分明显 + 接近最终场景;且**不能写 "transparent"**(生成器会画棋盘格),
# 也避免纯绿幕 #00FF00(残留绿边难去)。按场景关键词映射到几个安全的纯色底名。
_DEFAULT_BG = "medium-gray"
_SCENE_BG: dict[str, str] = {
    "forest": "dark-green", "森林": "dark-green", "jungle": "dark-green", "grass": "dark-green",
    "sky": "steel-blue", "天空": "steel-blue", "water": "steel-blue", "水": "steel-blue",
    "sea": "steel-blue", "ocean": "steel-blue",
    "dungeon": "dark-gray", "地牢": "dark-gray", "cave": "dark-gray", "洞": "dark-gray",
}


def pick_bg_color(scene_hint: str = "") -> str:
    """按场景提示挑一个安全的纯色背景名(用于参考帧生成 + 后续抠图基准)。

    Args:
        scene_hint: 场景描述 / 关键词(中英皆可);空 → 通用 ``medium-gray``。

    Returns:
        纯色背景名(如 ``dark-green`` / ``steel-blue`` / ``dark-gray`` / ``medium-gray``)。
        **绝不返回 "transparent" 或纯绿幕** —— 那两个会分别产生棋盘格与残留绿边。

    使用示例::

        pick_bg_color("a knight in a forest")  # -> "dark-green"
        pick_bg_color("underwater temple")     # -> "steel-blue"
        pick_bg_color("")                       # -> "medium-gray"
    """
    low = scene_hint.lower()
    for kw, color in _SCENE_BG.items():
        if kw in low:
            return color
    return _DEFAULT_BG


def build_reference_prompt(name: str, description: str, bg_color: str) -> str:
    """拼参考帧(人物模板)提示词。模板见手册 7.5。

    Args:
        name: 角色名。
        description: 外观描述(材质 / 配色 / 持物等)。
        bg_color: 纯色背景名(用 :func:`pick_bg_color` 得到)。

    Returns:
        完整提示词,含「纯色底 + 全身 + 中性站姿」这三条对后续 i2v / 抠图友好的约束。
    """
    subject = f"{name}, {description}".strip().strip(",")
    return (
        f"{subject}. Centered on a solid {bg_color} background. "
        "Full body, head to feet, standing neutral pose, no shadow."
    )


@dataclass
class SpriteActionSpec:
    """一个动作的生成规格(端到端 :meth:`SpritePipeline.run` 的输入项)。"""

    name: str                       # 动作名,用于目录 / 文件命名(如 "run")
    prompt: str                     # 动作描述(只描述动作变化,不重述外观)
    cyclic: bool = True             # 循环动作(走 / 跑 / 闲置)抽单周期闭环;一次性动作裁区间
    seconds: int = 0                # 视频时长(秒);0 = 按 cyclic 取默认(循环 3s / 一次性 2s)
    n_frames: int = 8               # 交付帧数
    kind: str = "swing"             # 仅一次性动作用:"swing"(挥砍)/ "airborne"(腾空)

    def video_seconds(self) -> int:
        """实际视频时长:显式给了用给的,否则循环 3s / 一次性 2s(手册 3.2 建议区间)。"""
        if self.seconds > 0:
            return self.seconds
        return 3 if self.cyclic else 2


@dataclass
class SpriteActionResult:
    """一个动作的产物:干净 PNG 帧序列 + 元信息。"""

    name: str
    frames: list[bytes] = field(default_factory=list)   # RGBA PNG,按播放序
    cyclic: bool = True
    cost_cents: int = 0


@dataclass
class SpriteCharacterResult:
    """一个角色的完整产物:参考帧 + 各动作序列 + 预估总成本。"""

    name: str
    reference: bytes = b""                                       # 参考帧 PNG
    actions: list[SpriteActionResult] = field(default_factory=list)
    total_cost_cents: int = 0


class _NullProgress(ProgressPort):
    """无回调时的占位;什么也不做(引擎不该因为没人听进度就改行为)。"""

    def step(self, stage: str, i: int, total: int, note: str = "") -> None:  # noqa: D102
        return None


class SpritePipeline:
    """精灵动画管线编排器:参考帧 → 动作帧 → 视频 → 帧 → 循环 → 干净 PNG。

    由外部注入三个 provider(可换后端 / 可注桩测试)。所有方法产出 **帧 bytes**,不落盘、
    不碰存储。默认抠图用 :class:`ColorMatteProvider`(纯色背景边缘更干净),见
    :func:`build_default_pipeline`。

    使用示例::

        pipe = build_default_pipeline()
        ref = pipe.generate_reference("Knight", "silver armor, blue cape", scene_hint="forest")
        pose = pipe.generate_action_frame(ref, "running forward")
        frames = pipe.animate(pose, "running forward", n_frames=8, cyclic=True)
        # frames: list[bytes],每个是一张抠好底的 RGBA PNG
    """

    def __init__(
        self,
        image: ImageProvider,
        video: VideoProvider,
        matte: MatteProvider,
        *,
        progress: ProgressPort | None = None,
        size: str = "1280x720",
    ) -> None:
        self._image = image
        self._video = video
        self._matte = matte
        self._progress = progress or _NullProgress()
        self._size = size

    # ── Step 1:参考帧 ────────────────────────────────────────────────────────
    def generate_reference(
        self,
        name: str,
        description: str,
        *,
        scene_hint: str = "",
        bg_color: str | None = None,
    ) -> bytes:
        """生成基础参考帧(人物模板)。文生图,纯色底,全身中性站姿。

        Args:
            name: 角色名。
            description: 外观描述。
            scene_hint: 场景提示,用于挑背景色(见 :func:`pick_bg_color`)。
            bg_color: 显式指定背景色名,覆盖 ``scene_hint`` 的推断。

        Returns:
            参考帧 PNG bytes。同一角色的所有动作都从这一张 image-to-image 派生,保证一致性。
        """
        color = bg_color or pick_bg_color(scene_hint)
        prompt = build_reference_prompt(name, description, color)
        self._progress.step("reference", 0, 1, f"参考帧: {name}(底 {color})")
        img = self._image.gen_image(prompt, [])
        self._progress.step("reference", 1, 1, "参考帧完成")
        return img

    # ── Step 2:动作帧(image-to-image)──────────────────────────────────────
    def generate_action_frame(self, reference: bytes, action_prompt: str) -> bytes:
        """从参考帧 image-to-image 生成动作帧。提示词**只描述动作变化,不重述外观**。

        Args:
            reference: 参考帧 PNG(:meth:`generate_reference` 的产物)。
            action_prompt: 动作描述,如 ``"running forward"`` / ``"attacking overhead"``。

        Returns:
            动作帧 PNG bytes(供 :meth:`animate` 作为视频首帧)。
        """
        self._progress.step("action_frame", 0, 1, f"动作帧: {action_prompt}")
        img = self._image.gen_image(action_prompt, [reference])
        self._progress.step("action_frame", 1, 1, "动作帧完成")
        return img

    # ── Step 3-6:视频 → 抽帧 → 循环检测 → 批量抠图 ─────────────────────────
    def animate(
        self,
        pose: bytes,
        action_prompt: str,
        *,
        n_frames: int = 8,
        cyclic: bool = True,
        seconds: int = 5,
        kind: str = "swing",
        sidecar: str | Path | None = None,
    ) -> list[bytes]:
        """从动作帧生成短视频,抽帧、找循环点、批量抠图,得到干净 PNG 序列。

        Args:
            pose: 动作帧 / 参考帧 PNG(作为视频首帧)。
            action_prompt: 视频动作描述。
            n_frames: 交付帧数。
            cyclic: 循环动作(走 / 跑 / 闲置)抽单周期闭环;一次性动作(攻击 / 死亡)裁动作区间。
            seconds: 视频时长(秒)。
            kind: 仅一次性动作用,``"swing"`` / ``"airborne"``。
            sidecar: 视频任务凭据落盘路径。给了它,视频超时(未失败)会保留凭据并抛
                :class:`~windup_framework.providers.VideoJobTimeout`,之后可用
                :meth:`resume_animation` 免费续跑;不给则超时即失败、无处恢复。

        Returns:
            长度 = ``n_frames`` 的干净 RGBA PNG 帧序列。

        Raises:
            VideoJobTimeout: 视频轮询超时且给了 ``sidecar``(可 resume)。
            ValueError: 抽帧 / 选帧得不到足够帧(视频太短或动作区间过窄)。
        """
        self._progress.step("video", 0, 4, f"i2v 生成 {seconds}s 视频: {action_prompt}")
        kwargs = {"seconds": seconds, "size": self._size}
        if sidecar is not None:
            kwargs["sidecar"] = sidecar          # 仅在需要 resume 时传,兼容不支持 sidecar 的 provider
        video = self._video.i2v(pose, action_prompt, **kwargs)
        return self._frames_from_video(video, n_frames=n_frames, cyclic=cyclic, kind=kind)

    def resume_animation(
        self,
        sidecar: str | Path,
        *,
        n_frames: int = 8,
        cyclic: bool = True,
        kind: str = "swing",
    ) -> list[bytes]:
        """对超时的视频任务免费续跑(见 :meth:`animate` 的 ``sidecar``),再走抽帧→循环→抠图。

        Raises:
            RuntimeError: 注入的 video provider 不支持 resume(没有 ``resume`` 方法)。
        """
        resume = getattr(self._video, "resume", None)
        if resume is None:
            raise RuntimeError(
                f"{type(self._video).__name__} 不支持 resume;只有能对已提交任务续轮询的 "
                "provider(如 SufyVideoProvider)才能恢复超时的视频任务"
            )
        self._progress.step("video", 0, 4, f"resume 续跑视频任务(sidecar={sidecar})")
        video = resume(sidecar)
        return self._frames_from_video(video, n_frames=n_frames, cyclic=cyclic, kind=kind)

    def _frames_from_video(
        self, video: bytes, *, n_frames: int, cyclic: bool, kind: str
    ) -> list[bytes]:
        """视频 bytes → 干净 PNG 序列(Step 4-6)。抽帧 / 选帧 / 抠图全部复用现有实现。"""
        self._progress.step("extract", 1, 4, "抽帧")
        dense = extract_all_frames_bytes(video)
        if not dense:
            raise ValueError("视频无可解码帧(生成异常或格式不支持)")

        if cyclic:
            self._progress.step("loop", 2, 4, f"循环检测,抽单周期 {n_frames} 帧(无缝 loop)")
            picked = pick_cycle(dense, n_frames)
        else:
            self._progress.step("loop", 2, 4, f"裁动作区间取 {n_frames} 帧(不闭环,{kind})")
            picked = pick_oneshot(dense, n_frames, kind=kind)

        self._progress.step("matte", 3, 4, f"批量抠图 {len(picked)} 帧")
        clean = [self._matte.cutout(_png(im)) for im in picked]
        self._progress.step("matte", 4, 4, f"完成 {len(clean)} 帧")
        return clean

    # ── 端到端 ────────────────────────────────────────────────────────────────
    def run(
        self,
        name: str,
        description: str,
        actions: list[SpriteActionSpec],
        *,
        scene_hint: str = "",
        bg_color: str | None = None,
        use_action_frame: bool = True,
    ) -> SpriteCharacterResult:
        """跑完一个角色的全部动作:参考帧 → (每个动作)动作帧 → 视频 → 干净序列。

        Args:
            name / description: 角色名与外观描述(用于参考帧)。
            actions: 动作规格列表。
            scene_hint / bg_color: 背景色策略(见 :meth:`generate_reference`)。
            use_action_frame: 每个动作是否先 image-to-image 出一张动作帧再喂 i2v(手册流程,
                一致性更好,每动作多约 7¢);``False`` 则直接用参考帧当首帧(省钱)。

        Returns:
            :class:`SpriteCharacterResult`,含参考帧、各动作干净序列、预估总成本(¢)。

        使用示例::

            result = pipe.run(
                "Knight", "silver armor, blue cape, longsword",
                actions=[
                    SpriteActionSpec("idle", "standing idle, subtle breathing", cyclic=True),
                    SpriteActionSpec("run", "running forward", cyclic=True),
                    SpriteActionSpec("attack", "overhead sword swing", cyclic=False, kind="swing"),
                ],
                scene_hint="forest",
            )
        """
        reference = self.generate_reference(name, description, scene_hint=scene_hint, bg_color=bg_color)
        result = SpriteCharacterResult(name=name, reference=reference)
        result.total_cost_cents += _COST_REFERENCE_CENTS

        for spec in actions:
            first_frame = reference
            cost = 0
            if use_action_frame:
                first_frame = self.generate_action_frame(reference, spec.prompt)
                cost += _COST_ACTION_CENTS
            secs = spec.video_seconds()
            frames = self.animate(
                first_frame, spec.prompt,
                n_frames=spec.n_frames, cyclic=spec.cyclic, seconds=secs, kind=spec.kind,
            )
            cost += secs * _COST_VIDEO_PER_SEC_CENTS
            result.actions.append(
                SpriteActionResult(name=spec.name, frames=frames, cyclic=spec.cyclic, cost_cents=cost)
            )
            result.total_cost_cents += cost
        return result

    def estimate_cost(self, actions: list[SpriteActionSpec], *, use_action_frame: bool = True) -> int:
        """预估一个角色跑完 ``actions`` 的成本(¢)。不含真实计费波动,仅按手册单价估。"""
        total = _COST_REFERENCE_CENTS
        for spec in actions:
            if use_action_frame:
                total += _COST_ACTION_CENTS
            total += spec.video_seconds() * _COST_VIDEO_PER_SEC_CENTS
        return total


def mirror_frames(frames: list[bytes]) -> list[bytes]:
    """水平翻转每一帧(左右朝向)。

    生成器空间感弱,"facing left" 与 "facing right" 往往生成成一样(手册 2.4)。所以只生成
    一个朝向,另一个朝向运行时**水平翻转**得到 —— 零成本、零 API。

    Args:
        frames: RGBA PNG 帧序列。

    Returns:
        逐帧水平镜像后的 PNG 序列(顺序不变,只翻像素)。
    """
    out = []
    for png in frames:
        im = _img(png).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        out.append(_png(im))
    return out


def build_default_pipeline(
    *, progress: ProgressPort | None = None, size: str = "1280x720", video_model: str | None = None
) -> SpritePipeline:
    """用仓库现役 provider 组装默认管线(SUFY 图像 / 视频 + Color-Matting 抠图)。

    provider 惰性构造(其内部 httpx / onnx 依赖各自惰性),故 import 本模块零成本。
    抠图默认 :class:`ColorMatteProvider`:精灵管线的背景是刻意生成的纯色,物理 Color Matting
    的边缘比通用显著性抠图干净。
    """
    from windup_framework.providers import ColorMatteProvider, SufyImageProvider, SufyVideoProvider

    return SpritePipeline(
        image=SufyImageProvider(),
        video=SufyVideoProvider(model=video_model),
        matte=ColorMatteProvider(),
        progress=progress,
        size=size,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
class _StderrProgress(ProgressPort):
    """CLI 用:进度打到 stderr(stdout 留给结构化 JSON)。"""

    def step(self, stage: str, i: int, total: int, note: str = "") -> None:
        import sys

        print(f"▶ [{stage}] {i}/{total} {note}", file=sys.stderr)


def _write_frames(frames: list[bytes], out_dir: Path) -> list[str]:
    """把帧序列写成 ``0001.png`` 起的编号 PNG,返回路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, png in enumerate(frames, start=1):
        p = out_dir / f"{i:04d}.png"
        p.write_bytes(png)
        paths.append(str(p))
    return paths


def _main() -> int:
    import argparse
    import json

    from windup_common.result import Response
    from windup_framework.providers import VideoJobTimeout

    parser = argparse.ArgumentParser(
        prog="windup_ai_engine.sprite_pipeline", description="精灵动画视频管线(端到端编排)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ref = sub.add_parser("reference", help="Step 1:生成参考帧(人物模板)")
    p_ref.add_argument("--name", required=True)
    p_ref.add_argument("--description", required=True)
    p_ref.add_argument("--scene", default="", help="场景提示,用于挑背景色")
    p_ref.add_argument("--bg-color", default=None, help="显式背景色名,覆盖 --scene 推断")
    p_ref.add_argument("-o", "--output", required=True, help="输出 PNG 路径")

    p_act = sub.add_parser("action", help="Step 2:从参考帧 image-to-image 生成动作帧")
    p_act.add_argument("--reference", required=True, help="参考帧 PNG 路径")
    p_act.add_argument("--prompt", required=True, help="动作描述(只描述动作,不重述外观)")
    p_act.add_argument("-o", "--output", required=True, help="输出 PNG 路径")

    p_ani = sub.add_parser("animate", help="Step 3-6:视频→抽帧→循环检测→批量抠图")
    p_ani.add_argument("--pose", help="动作帧 PNG 路径(非 --resume 时必填)")
    p_ani.add_argument("--prompt", default="", help="视频动作描述")
    p_ani.add_argument("--frames", type=int, default=8, help="交付帧数")
    grp = p_ani.add_mutually_exclusive_group()
    grp.add_argument("--cyclic", dest="cyclic", action="store_true", help="循环动作(默认)")
    grp.add_argument("--oneshot", dest="cyclic", action="store_false", help="一次性动作(裁区间)")
    p_ani.set_defaults(cyclic=True)
    p_ani.add_argument("--seconds", type=int, default=3, help="视频时长(秒)")
    p_ani.add_argument("--kind", default="swing", choices=("swing", "airborne"), help="一次性动作抽帧判据")
    p_ani.add_argument("--sidecar", default=None, help="视频任务凭据落盘路径(超时可 resume)")
    p_ani.add_argument("--resume", action="store_true", help="对 --sidecar 记录的任务免费续跑")
    p_ani.add_argument("-o", "--output", required=True, help="输出目录(写 0001.png ...)")

    p_run = sub.add_parser("run", help="端到端:一个角色跑完全部动作")
    p_run.add_argument("--spec", required=True, help="角色规格 JSON 文件路径")
    p_run.add_argument("-o", "--output", required=True, help="输出目录")

    args = parser.parse_args()
    progress = _StderrProgress()

    try:
        if args.cmd == "reference":
            pipe = build_default_pipeline(progress=progress)
            data = pipe.generate_reference(
                args.name, args.description, scene_hint=args.scene, bg_color=args.bg_color
            )
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            print(Response.success({"path": str(out)}).model_dump_json())
            return 0

        if args.cmd == "action":
            pipe = build_default_pipeline(progress=progress)
            data = pipe.generate_action_frame(Path(args.reference).read_bytes(), args.prompt)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            print(Response.success({"path": str(out)}).model_dump_json())
            return 0

        if args.cmd == "animate":
            pipe = build_default_pipeline(progress=progress)
            try:
                if args.resume:
                    if not args.sidecar:
                        print(Response.fail("--resume 需要 --sidecar", code=400).model_dump_json())
                        return 2
                    frames = pipe.resume_animation(
                        args.sidecar, n_frames=args.frames, cyclic=args.cyclic, kind=args.kind
                    )
                else:
                    if not args.pose:
                        print(Response.fail("animate 需要 --pose(或 --resume)", code=400).model_dump_json())
                        return 2
                    frames = pipe.animate(
                        Path(args.pose).read_bytes(), args.prompt,
                        n_frames=args.frames, cyclic=args.cyclic, seconds=args.seconds,
                        kind=args.kind, sidecar=args.sidecar,
                    )
            except VideoJobTimeout as exc:
                # 超时不是失败:凭据已留在 sidecar,提示可 resume。用 503 表达「服务未给结论」。
                print(Response.fail(
                    f"{exc};稍后用 --resume --sidecar {args.sidecar} 续跑", code=503,
                    data={"job_id": exc.job_id, "sidecar": args.sidecar},
                ).model_dump_json())
                return 3
            paths = _write_frames(frames, Path(args.output))
            print(Response.success({"frame_count": len(paths), "paths": paths}).model_dump_json())
            return 0

        # run
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        actions = [
            SpriteActionSpec(
                name=a["name"], prompt=a["prompt"],
                cyclic=bool(a.get("cyclic", True)), seconds=int(a.get("seconds", 0)),
                n_frames=int(a.get("n_frames", 8)), kind=a.get("kind", "swing"),
            )
            for a in spec.get("actions", [])
        ]
        pipe = build_default_pipeline(progress=progress)
        result = pipe.run(
            spec["name"], spec["description"], actions,
            scene_hint=spec.get("scene_hint", ""), bg_color=spec.get("bg_color"),
            use_action_frame=bool(spec.get("use_action_frame", True)),
        )
        out_dir = Path(args.output)
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "reference.png").write_bytes(result.reference)
        summary = {"name": result.name, "reference": str(out_dir / "reference.png"),
                   "cost_cents": result.total_cost_cents, "actions": []}
        for act in result.actions:
            paths = _write_frames(act.frames, out_dir / act.name)
            summary["actions"].append({"name": act.name, "frame_count": len(paths), "cost_cents": act.cost_cents})
        print(Response.success(summary).model_dump_json())
        return 0
    except VideoJobTimeout as exc:
        print(Response.fail(str(exc), code=503, data={"job_id": exc.job_id}).model_dump_json())
        return 3
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(Response.fail(str(exc), code=500).model_dump_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
