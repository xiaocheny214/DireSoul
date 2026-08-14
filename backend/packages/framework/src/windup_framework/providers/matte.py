"""主体抠图 MatteProvider —— onnxruntime 直跑 u2netp,不依赖 rembg。

为什么不用 rembg:rembg → pymatting → numba 0.53 / llvmlite 0.36 这条老链在 Python
3.12 无轮子(实测装不上)。而 rembg 内核就是"u2netp.onnx 过一遍 onnxruntime";默认
``alpha_matting=False`` 时根本不碰 pymatting。故直调 onnxruntime,甩掉整条死重依赖,
3.12 干净可装、可进 lock。同模型(u2netp),同质量。

模型解析顺序:显式 ``model_path`` → 缓存目录已存在 → 从 ``model_url`` 惰性下载。
onnxruntime 惰性导入(启动慢、按需加载),会话按需构建一次。
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .interfaces import MatteProvider

# u2netp:轻量版(~4.7MB)。rembg 官方 release 托管;国内不可达时可预置 model_path。
_U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
_DEFAULT_CACHE = Path.home() / ".cache" / "windup" / "u2netp.onnx"

# u2net 预处理常量(与 rembg 一致)。
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_SIZE = (320, 320)


# 只清理"几乎精确等于底色"的像素。阈值必须窄:2026-08-07 实测,一个铁锈橙毛
# (222,130,70)的角色配玫红底(222,41,124),两者红通道完全相同、欧氏距离仅 104 ——
# 宽阈值会把毛判成半透明并去"反解",越解越坏(先成橄榄绿再成亮绿)。橙毛 d≈117,
# 阈值 38 完全碰不到它;而闭合空隙里的背景 d≈0,能干净移除。
_KEY_KILL = 38.0    # d < 此值 → 判为纯背景
_KEY_SOFT = 14.0    # 到 _KEY_KILL + _KEY_SOFT 之间线性过渡,避免硬边锯齿
_BG_FLAT_STD = 8.0  # 四角色标准差上限;超过说明底不是纯色,不做任何清理

# 采样前先丢掉最外圈像素。视频帧的最外一两行/列常是**编码器边缘伪影**,不是底色:
# 2026-08-10 实测 9 段真 i2v 视频 × 16 帧 = 144 帧,贴边采样时 26 帧(18%)判"底不均匀"
# 而跳过清理,逐一查证全部由最外圈造成 —— 白底母版视频最右一列整列纯黑(std 50.4),
# 待机视频最顶一行偏暗(std 8.4,恰好压线越过 8)。往里让 1 px 就降到 1.9、让 2 px 降到 1.88,
# 144 帧零误跳;三张静态母版的取样中位色一个字节都没变(220/64/135、222/39/130、222/41/124)。
# 取 2 是为容下 2 px 宽的边框;真正不均匀的底(噪声/渐变/拼色)让多少都照样超阈值,守卫不松。
_EDGE_SKIP = 2
_CORNER = 12        # 每个角的采样块边长


def _corner_pixels(rgb: np.ndarray) -> np.ndarray:
    """四角采样块(跳过最外圈 ``_EDGE_SKIP`` 像素)拼成的 (N, 3) 像素表。

    图太小时(四角会互相重叠)不让,退回贴边取 —— 合成测试图和缩略图走这条路。
    """
    k = _CORNER
    s = _EDGE_SKIP if min(rgb.shape[:2]) > 2 * (_EDGE_SKIP + k) else 0
    r = rgb[s : rgb.shape[0] - s, s : rgb.shape[1] - s] if s else rgb
    return np.concatenate([
        r[:k, :k].reshape(-1, 3), r[:k, -k:].reshape(-1, 3),
        r[-k:, :k].reshape(-1, 3), r[-k:, -k:].reshape(-1, 3),
    ])


# 空洞填充用。_HOLE_ALPHA:低于此 alpha 才算"透明",参与空洞判定。
# _HOLE_BG_TOL:到底色的距离低于此值 → 判为"确实是底色"。取值依据(2026-08-11 实测,
# 1280×720 真实视频帧):纯背景区域的色距 p99.9≈6.5、最大 11.1(视频压缩噪点);
# 而被误杀的浅肤色像素连通域中位色距 ≥17.1。14 落在这条 1.5 倍间隙里。
_HOLE_ALPHA = 0.03
_HOLE_BG_TOL = 14.0


def _bg_key(rgb: np.ndarray) -> np.ndarray | None:
    """四角取样估底色 key;底不够均匀(std 超阈值)时返回 None = 不做任何基于底色的判断。

    抽成独立函数是为了让"底色是什么"只有一个真相源 —— 键控清理(``_flat_bg_penalty``)
    和空洞填充(``_fill_enclosed_holes``)必须按同一个 key 判断,否则一个把某块当背景
    清掉、另一个又把它当主体填回来,互相打架。取样统一走 :func:`_corner_pixels`,
    连"跳过最外圈编码器伪影"这条也只有一份实现。
    """
    corners = _corner_pixels(rgb)          # 跳过编码器边缘伪影,见 _EDGE_SKIP
    if float(corners.std(axis=0).max()) > _BG_FLAT_STD:
        return None
    return np.median(corners, axis=0).astype(np.float32)


def _spread(seed: np.ndarray, region: np.ndarray) -> np.ndarray:
    """在 ``region`` 内从 ``seed`` 出发做 4-邻接连通扩散,返回可达集合。

    为什么不写逐像素 BFS:交付前的帧是 1280×720(约 92 万像素),纯 Python BFS 要几十秒,
    抠图是逐帧调用的,扛不住。这里按**行/列游程**传播 —— 一个 pass 就能把可达性推过
    整条连续游程(距离不限),而不是每 pass 只推进一个像素,真实角色轮廓几个 pass 收敛。

    同一行里被非 region 像素隔断的两段游程,``cumsum(~region)`` 必然取到不同的 id,
    因此可以用 ``bincount`` 一次算出"每条游程里有没有种子"。
    """
    reach = seed & region
    while True:
        before = int(reach.sum())
        for transposed in (False, True):
            reg = region.T if transposed else region
            rch = reach.T if transposed else reach
            rows, cols = reg.shape
            run = np.cumsum(~reg, axis=1)
            keys = run + np.arange(rows)[:, None] * (cols + 1)
            hit = np.bincount(keys[rch], minlength=rows * (cols + 1)) > 0
            new = reg & hit[keys]
            reach = new.T if transposed else new
        if int(reach.sum()) == before:
            return reach


def _fill_enclosed_holes(alpha: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """把"被主体围住、且整块都不是底色"的透明连通域填回主体(alpha=1)。

    要解决的问题:u2netp 判错或键控误杀会在主体内部留下透明洞,放大看是背景直接透出来。

    **为什么只判"不与边界连通"不够 —— 会把两腿之间填实。** 直觉上腿间空隙从下方通到
    画面底边,所以"从边界出发的连通域"就能保护它。2026-08-11 在真实走路帧上实测:
    **不成立**。迈步相里两只靴子在下方交叠,把腿间空隙彻底封死 —— 它就是一块不与边界
    连通的背景域(实测 src_017 有 530 像素、归档 frame_03 有 129 像素),只按连通性判,
    这一整块会被填成主体,两条腿直接焊在一起。

    所以判据是**连通性 + 颜色**两条一起:一个透明连通域只要"碰到画面边界"或者"里面
    存在任何一个确实是底色的像素",就不是洞。腿间空隙整块就是底色(实测中位色距 6.2,
    远低于 _HOLE_BG_TOL),必然被这条否决;而被误杀的主体像素(实测中位色距 ≥17.1)
    不含底色像素,才会被填。两条否决合成一次扩散:种子 = 边界上的透明像素 ∪ 底色像素。

    与 ``_flat_bg_penalty`` 的分工:那个函数按颜色**做减法**(把闭合空隙里的底色清掉),
    这个函数按颜色**决定不加回来** —— 同一个 key、同一个方向,不会互相拆台。
    """
    key = _bg_key(rgb)
    if key is None:
        return alpha                      # 底不是纯色 → 无从判断哪块是真空隙,一律不填
    transparent = alpha < _HOLE_ALPHA
    if not transparent.any():
        return alpha
    border = np.zeros_like(transparent)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    is_bg_color = np.linalg.norm(rgb - key, axis=2) < _HOLE_BG_TOL
    seed = transparent & (border | is_bg_color)
    holes = transparent & ~_spread(seed, transparent)
    if not holes.any():
        return alpha
    out = alpha.copy()
    out[holes] = 1.0
    return out


def _flat_bg_penalty(rgb: np.ndarray) -> np.ndarray:
    """底色清理系数(0=纯背景,1=主体),形状与图同宽高。

    为什么需要它:u2netp 是显著性模型,对**闭合区域**天然失灵 —— 四足角色腿间的
    背景是一块被主体围住的空隙,显著性把它当成主体内部,整块底色留在产物里
    (2026-08-07 实测)。而母版底色是刻意生成的纯色,均匀度极高(实测四角标准差 1.0~1.2),
    用它做一次窄阈值清理就能补上这个洞。

    与"按颜色抠是死路"那条规则的边界:那条说的是**拿颜色当主体判据**(白底浅色角色
    会被抠穿)。这里主体判据仍然是 u2netp,颜色只用来**做减法** —— 绝不新增主体像素,
    最坏情况是少清理一点,不会抠穿角色。底色不够均匀时(std 超阈值)直接返回全 1,
    等于不清理。

    **逐帧独立采样是安全的**(2026-08-10 在真视频帧上验证):同一段视频里逐帧算出的 key 色
    几乎不动(9 段 i2v 实测帧间位移 <= 1.73/255),故不需要跨帧共享一次采样。序列帧真正的
    闪烁源是**守卫在序列中途翻转**(部分帧清、部分帧不清):待机那段 16 帧里前 6 帧清、后 10 帧
    不清,主体面积逐帧变化 CV 从 0.0036 跳到 0.0197、第 6 帧单帧跳 4.25%。跳过最外圈后
    守卫不再翻转,CV 回到 0.0028 —— 比完全不清理还稳(清理同时抹掉了会自己抖的底色描边)。
    """
    key = _bg_key(rgb)
    if key is None:
        return np.ones(rgb.shape[:2], dtype=np.float32)   # 底不是纯色 → 不动
    d = np.linalg.norm(rgb - key, axis=2)
    return np.clip((d - _KEY_KILL) / _KEY_SOFT, 0.0, 1.0).astype(np.float32)


class OnnxU2NetMatteProvider(MatteProvider):
    """u2netp.onnx via onnxruntime。frame bytes → 抠好的 PNG(RGBA) bytes。"""

    def __init__(self, model_path: str | Path | None = None, model_url: str = _U2NETP_URL) -> None:
        self._model_path = Path(model_path) if model_path else _DEFAULT_CACHE
        self._model_url = model_url
        self._session = None  # 惰性

    def _ensure_model(self) -> Path:
        if not self._model_path.exists():
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self._model_url, self._model_path)
        return self._model_path

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort  # 惰性:导入慢
            except ImportError as e:       # pragma: no cover - 取决于安装环境
                # **不静默降级。** 这里曾在 ImportError 时回落到"取四角主色做 chroma-key",
                # 有两个问题:①猜背景色 —— 白底母版四角就是白色,浅色角色(骨白/银甲)与背景
                # 撞色会被抠穿;②静默 —— 开发机上看着能跑、输出其实是坏的,要到产物验收才发现。
                raise RuntimeError(
                    "onnxruntime 不可用，无法做主体抠图。请安装 onnxruntime"
                    "（注意 <1.24 才有 macOS Intel 轮子）。"
                ) from e
            self._session = ort.InferenceSession(
                str(self._ensure_model()), providers=["CPUExecutionProvider"]
            )
        return self._session

    def soft_mask(self, img: Image.Image) -> np.ndarray:
        """u2netp 软掩码,原图尺寸的 ``[0, 1]`` float ndarray(0=背景,1=主体)。

        暴露它是为了让 :class:`ColorMatteProvider` 复用同一份 onnx 会话与前向,而不是
        另起一套推理 —— 「主体掩码是什么」只该有一个真相源(同 :mod:`.._subject` 的理由)。
        """
        return np.asarray(self._predict_mask(img), dtype=np.float32) / 255.0

    def _predict_mask(self, img: Image.Image) -> Image.Image:
        """u2netp 前向 → 单通道显著性 mask(L,原图尺寸)。"""
        im = img.convert("RGB").resize(_SIZE, Image.LANCZOS)
        ary = np.array(im).astype(np.float32)
        ary = ary / max(float(ary.max()), 1e-6)
        tmp = np.zeros((_SIZE[1], _SIZE[0], 3), dtype=np.float32)
        for c in range(3):
            tmp[:, :, c] = (ary[:, :, c] - _MEAN[c]) / _STD[c]
        tensor = np.expand_dims(tmp.transpose(2, 0, 1), 0).astype(np.float32)

        session = self._get_session()
        pred = session.run(None, {session.get_inputs()[0].name: tensor})[0][:, 0, :, :]
        mi, ma = float(pred.min()), float(pred.max())
        pred = (pred - mi) / max(ma - mi, 1e-6)
        mask = (pred.squeeze() * 255).astype(np.uint8)
        return Image.fromarray(mask, "L").resize(img.size, Image.LANCZOS)

    def cutout(self, frame: bytes) -> bytes:
        img = Image.open(io.BytesIO(frame)).convert("RGBA")
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        alpha = np.asarray(self._predict_mask(img), dtype=np.float32) / 255.0
        alpha = alpha * _flat_bg_penalty(rgb)
        alpha = _fill_enclosed_holes(alpha, rgb)
        out = np.dstack([np.asarray(img.convert("RGB")), alpha * 255.0]).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(out, "RGBA").save(buf, "PNG")
        return buf.getvalue()


# ── Color-Matting + u2netp 混合抠图(纯色背景专用)────────────────────────────
#
# 与 OnnxU2NetMatteProvider 的分工:那个是「显著性抠主体」,判据是 u2netp,颜色只在闭合
# 空隙里**做减法**;适合背景未知 / 非纯色的通用帧。本 provider 反过来:精灵管线的背景是
# **刻意生成的纯色底**(见 asset-workflow 参考手册 2.1「不要写 transparent」),此时用合成
# 方程 pixel = α·fg + (1-α)·bg 能从物理上算出 α 的下限,把半透明边缘、发丝这类 u2netp 会
# 一刀切硬边的地方还原成软过渡,并**反合成消掉被底色污染的边缘像素**(去彩色光环 fringing)。
# u2netp 软掩码只用来判「这块到底算不算主体」(三态 regime),不单独决定 α。

# 三态判定阈值(mask 前景占比)。低于 MIN → mask 失败,只信 Color Matting;高于 MAX →
# mask 把背景也圈进来了(泄漏),转自适应;之间 → 信 mask。与 asset-workflow 手册 5.2 一致。
_MASK_MIN_PCT = 5.0
_MASK_MAX_PCT = 70.0
_MASK_MIN_PX = 100

# 各 regime 的默认阈值(bg_thresh 越低越激进去背景;fg_thresh 越高越保护前景)。
_REGIME_DEFAULTS: dict[str, dict[str, float]] = {
    "trust": {"bg_thresh": 0.05, "fg_thresh": 1.0},   # 信 mask:mask 前景一律保留
    "adapt": {"bg_thresh": 0.05, "fg_thresh": 0.20},  # 自适应:阈值由 mask 值插值
    "color": {"bg_thresh": 0.10, "fg_thresh": 0.10},  # 只有 Color Matting
}
_REGIMES = ("auto", "trust", "adapt", "color")


def sample_bg_color(img: np.ndarray, block: int = 2) -> np.ndarray:
    """四角各取 ``block×block`` 块的平均色作为背景色。

    Args:
        img: RGB 图像 ``(H, W, 3)`` float,取值 ``[0, 1]``。
        block: 角落采样块边长。

    Returns:
        ``(3,)`` 平均背景色。

    取四角是「纯色底四角必是背景」这条几何假设;主体一般不顶到四个角。取平均(不是中位)
    是因为块本身已很小、且这里要的是 Color Matting 的基准色,轻微噪声用均值即可。
    """
    corners = np.concatenate([
        img[:block, :block].reshape(-1, 3),
        img[:block, -block:].reshape(-1, 3),
        img[-block:, :block].reshape(-1, 3),
        img[-block:, -block:].reshape(-1, 3),
    ])
    return corners.mean(axis=0)


def compute_alpha_color(img: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    """由合成方程反解每个像素 α 的**物理下限**。

    合成方程 ``pixel_c = α·fg_c + (1-α)·bg_c``。对未知 fg 取两个极端(fg=1 与 fg=0)分别得
    ``α >= (pixel_c - bg_c)/(1 - bg_c)`` 与 ``α >= (bg_c - pixel_c)/bg_c``,逐通道取最大即下限。

    Args:
        img: RGB ``(H, W, 3)`` float ``[0, 1]``。
        bg_color: ``(3,)`` 背景色 ``[0, 1]``。

    Returns:
        ``(H, W)`` α 下限,``[0, 1]``。

    分母接近 0(背景某通道接近 0 或 1)时该通道贡献不可靠,用 0.05 的余量跳过,避免除爆。
    """
    diff = img - bg_color[None, None, :]
    alpha = np.zeros(img.shape[:2], dtype=np.float64)
    for c in range(3):
        if 1.0 - bg_color[c] > 0.05:
            alpha = np.maximum(alpha, np.maximum(diff[:, :, c], 0) / (1.0 - bg_color[c]))
        if bg_color[c] > 0.05:
            alpha = np.maximum(alpha, np.maximum(-diff[:, :, c], 0) / bg_color[c])
    return np.clip(alpha, 0.0, 1.0)


def recover_foreground(img: np.ndarray, alpha: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    """反合成还原前景真实色 ``fg = (pixel - (1-α)·bg) / α``,消除边缘的背景色残留。

    Args:
        img: RGB ``(H, W, 3)`` float ``[0, 1]``。
        alpha: ``(H, W)`` ``[0, 1]``。
        bg_color: ``(3,)``。

    Returns:
        还原后的前景 ``(H, W, 3)`` ``[0, 1]``。

    α 接近 0 处除法会放大噪声,故 α<0.02 的像素直接判黑(它们几乎全透明,颜色不会被看到)。
    """
    a = alpha[:, :, np.newaxis]
    bg = bg_color[np.newaxis, np.newaxis, :]
    safe_a = np.where(a > 0.02, a, 1.0)
    fg = np.clip((img - (1.0 - a) * bg) / safe_a, 0.0, 1.0)
    fg[alpha < 0.02] = 0.0
    return fg


def detect_regime(mask_soft: np.ndarray) -> str:
    """按 u2netp 软掩码的前景覆盖率选抠图模式:``trust`` / ``adapt`` / ``color``。"""
    mask_fg = int((mask_soft > 0.5).sum())
    pct = mask_fg / mask_soft.size * 100
    if mask_fg < _MASK_MIN_PX or pct < _MASK_MIN_PCT:
        return "color"
    if pct > _MASK_MAX_PCT:
        return "adapt"
    return "trust"


class ColorMatteProvider(MatteProvider):
    """纯色背景专用的 Color-Matting + u2netp 混合抠图。frame bytes → RGBA PNG bytes。

    实现 :class:`~.interfaces.MatteProvider`,可与 :class:`OnnxU2NetMatteProvider` 互换注入。
    复用后者的 onnx 会话取软掩码(不另起推理);α 由 Color Matting 的物理下限主导,mask 只
    决定三态 regime。适用于精灵管线这类**背景是刻意生成的纯色**的场景。

    使用示例::

        provider = ColorMatteProvider()
        clean_png = provider.cutout(frame_png)          # RGBA,边缘干净
        qa_png = provider.preview(frame_png)            # 对比色背景合成图,肉眼查透明度
    """

    def __init__(
        self,
        mask_provider: OnnxU2NetMatteProvider | None = None,
        *,
        regime: str = "auto",
        bg_thresh: float | None = None,
        fg_thresh: float | None = None,
    ) -> None:
        if regime not in _REGIMES:
            raise ValueError(f"regime 须是 {_REGIMES} 之一,收到 {regime!r}")
        # 默认自建一个 u2netp provider 取软掩码;传入可复用已加载的会话(批量模式省一次装载)。
        self._mask = mask_provider or OnnxU2NetMatteProvider()
        self._regime = regime
        self._bg_thresh = bg_thresh
        self._fg_thresh = fg_thresh

    def _rgba(self, frame: bytes) -> tuple[np.ndarray, np.ndarray]:
        """解码 → (RGB float[0,1], u2netp 软掩码[0,1])。两者同尺寸。"""
        img = Image.open(io.BytesIO(frame)).convert("RGB")
        rgb = np.asarray(img, dtype=np.float64) / 255.0
        mask = self._mask.soft_mask(img).astype(np.float64)
        return rgb, mask

    def _matte(self, rgb: np.ndarray, mask_soft: np.ndarray,
               bg_color_override: np.ndarray | None = None) -> np.ndarray:
        """核心抠图:RGB + 软掩码 → RGBA uint8。见类 docstring 的算法说明。"""
        bg_color = bg_color_override if bg_color_override is not None else sample_bg_color(rgb)
        alpha_color = compute_alpha_color(rgb, bg_color)

        regime = detect_regime(mask_soft) if self._regime == "auto" else self._regime
        d = _REGIME_DEFAULTS[regime]
        bt = self._bg_thresh if self._bg_thresh is not None else d["bg_thresh"]
        ft = self._fg_thresh if self._fg_thresh is not None else d["fg_thresh"]

        if regime == "color":
            alpha = alpha_color
        elif regime == "trust":
            # 信 mask:mask 判为前景(>0.05)的地方 α 至少取到 mask 值,从不因 Color Matting 抹掉
            is_bg = (alpha_color < bt) | (mask_soft < 0.05)
            alpha = np.where(is_bg, alpha_color, np.maximum(alpha_color, mask_soft))
        else:  # adapt:阈值随 mask 值在 [bt, ft] 间插值,mask 前景也可被判背景(治泄漏)
            thresh = bt + mask_soft * (ft - bt)
            is_bg = alpha_color < thresh
            alpha = np.where(is_bg, alpha_color, np.maximum(alpha_color, mask_soft))

        alpha = alpha.copy()
        alpha[alpha < 0.01] = 0.0
        fg = recover_foreground(rgb, alpha, bg_color)

        h, w = rgb.shape[:2]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[:, :, :3] = (fg * 255).clip(0, 255).astype(np.uint8)
        out[:, :, 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
        return out

    def cutout(self, frame: bytes) -> bytes:
        """移除纯色背景,返回 RGBA PNG bytes(实现 MatteProvider 契约)。"""
        rgb, mask = self._rgba(frame)
        out = self._matte(rgb, mask)
        buf = io.BytesIO()
        Image.fromarray(out, "RGBA").save(buf, "PNG")
        return buf.getvalue()

    def preview(self, frame: bytes, bg: tuple[int, int, int] = (255, 0, 255)) -> bytes:
        """QA 预览:把抠图结果合成到对比色(默认品红)背景上,返回 RGB PNG bytes。

        肉眼查透明度的唯一可靠方式 —— 从裸 RGBA 看不出边缘残留 / 抠穿,合成到对比色上一眼可见。
        """
        rgb, mask = self._rgba(frame)
        rgba = self._matte(rgb, mask)
        cut = Image.fromarray(rgba, "RGBA")
        canvas = Image.new("RGBA", cut.size, (*bg, 255))
        canvas.alpha_composite(cut)
        buf = io.BytesIO()
        canvas.convert("RGB").save(buf, "PNG")
        return buf.getvalue()


def _main() -> int:
    """CLI:单帧 / 批量抠纯色背景,输出统一 ``Response`` 形状 JSON。"""
    import argparse
    from pathlib import Path

    from windup_common.result import Response

    p = argparse.ArgumentParser(description="Color-Matting + u2netp 混合抠图(纯色背景)")
    p.add_argument("input", nargs="?", help="单帧输入图路径(与 --batch 二选一)")
    p.add_argument("-o", "--output", required=True, help="单帧输出 PNG 路径,或 --batch 时的输出目录")
    p.add_argument("--batch", metavar="DIR", help="批量模式:处理目录下所有 PNG(BiRefNet/u2netp 只加载一次)")
    p.add_argument("-m", "--mode", default="auto", choices=_REGIMES, help="抠图模式(默认 auto)")
    p.add_argument("--bg-thresh", type=float, default=None, help="背景阈值(越低越激进)")
    p.add_argument("--fg-thresh", type=float, default=None, help="前景阈值(越高越保护)")
    p.add_argument("--preview", action="store_true", help="额外生成 _qa.png(对比色背景合成,查透明度)")
    args = p.parse_args()

    provider = ColorMatteProvider(regime=args.mode, bg_thresh=args.bg_thresh, fg_thresh=args.fg_thresh)

    try:
        if args.batch:
            src_dir, out_dir = Path(args.batch), Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            done = []
            for src in sorted(src_dir.glob("*.png")):
                (out_dir / src.name).write_bytes(provider.cutout(src.read_bytes()))
                if args.preview:
                    (out_dir / f"{src.stem}_qa.png").write_bytes(provider.preview(src.read_bytes()))
                done.append(str(out_dir / src.name))
            print(Response.success({"count": len(done), "paths": done}).model_dump_json())
            return 0

        if not args.input:
            print(Response.fail("单帧模式需给 input(或用 --batch)", code=400).model_dump_json())
            return 2
        data = Path(args.input).read_bytes()
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(provider.cutout(data))
        result = {"path": str(out_path)}
        if args.preview:
            qa = out_path.with_name(f"{out_path.stem}_qa.png")
            qa.write_bytes(provider.preview(data))
            result["qa"] = str(qa)
        print(Response.success(result).model_dump_json())
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(Response.fail(str(exc), code=500).model_dump_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
