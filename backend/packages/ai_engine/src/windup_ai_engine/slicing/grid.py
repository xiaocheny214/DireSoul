"""网格切分:把一张「kit 图」(网格里画多个物体)切成独立 PNG。

用途(参考 godogen asset-gen 的 grid_slice):生成一张 1K 图里画 2×2 四个道具,比逐个
生成更便宜、风格更一致,再按网格切成独立资产。与 :mod:`.extract`(视频→帧)是两码事——
那个按时间轴切,这个按空间网格切,所以放在 slicing 层但各自独立。

**引擎层零存储约定**:核心函数只吃 / 吐 ``bytes``(见 :mod:`.._imgio` 的同款理由),
落盘只发生在 CLI 边界。这样 server 侧要切图时拿字节直接上传对象存储,不经过本地临时文件。

CLI::

    python -m windup_ai_engine.slicing.grid input.png -o out/ --grid 2x2 --names sword,shield,potion,helm
"""
from __future__ import annotations

import io

from PIL import Image

__all__ = ["slice_grid_bytes"]


def slice_grid_bytes(
    src: bytes,
    cols: int,
    rows: int,
    names: list[str] | None = None,
) -> list[tuple[str, bytes]]:
    """把网格图 bytes 切成 ``cols × rows`` 个格子,返回 ``(名字, PNG bytes)`` 列表。

    Args:
        src: 输入网格图 PNG/JPG bytes。
        cols: 列数(横向格子数),须 >= 1。
        rows: 行数(纵向格子数),须 >= 1。
        names: 每格名字(不含扩展名),长度必须 = ``cols * rows``;
            不给则用 ``01`` / ``02`` / ... 顺序命名(行优先)。

    Returns:
        长度 = ``cols * rows`` 的列表,按**行优先**顺序(左上→右,再下一行),
        每项是 ``(name, rgba_png_bytes)``。

    Raises:
        ValueError: ``cols`` / ``rows`` < 1,或 ``names`` 长度与格子数不符。

    为什么切成 RGBA:kit 图常带透明底(或后续要抠图),统一 RGBA 免得某些格子丢 alpha。
    整除取格子尺寸(``w // cols``):非整除时右 / 下边余下的几像素被丢弃,好过把接缝
    的半像素塞进相邻格子造成串色 —— 生成 kit 图时本就该按整齐网格构图。

    使用示例::

        cells = slice_grid_bytes(png, 2, 2, names=["sword", "shield", "potion", "helm"])
        for name, data in cells:
            store.upload(f"{name}.png", data)
    """
    if cols < 1 or rows < 1:
        raise ValueError(f"网格必须至少 1×1,收到 {cols}×{rows}")
    total = cols * rows
    if names is not None and len(names) != total:
        raise ValueError(f"names 有 {len(names)} 个,网格是 {total} 格,数量必须一致")

    img = Image.open(io.BytesIO(src)).convert("RGBA")
    w, h = img.size
    cell_w, cell_h = w // cols, h // rows
    if cell_w < 1 or cell_h < 1:
        raise ValueError(
            f"图像 {w}×{h} 切成 {cols}×{rows} 后单格尺寸 {cell_w}×{cell_h} 非法;网格比图还大"
        )

    out: list[tuple[str, bytes]] = []
    for i in range(total):
        # 行优先:divmod 先得行号再得列号,与 names 的自然阅读顺序一致
        row, col = divmod(i, cols)
        x0, y0 = col * cell_w, row * cell_h
        cell = img.crop((x0, y0, x0 + cell_w, y0 + cell_h))
        name = names[i] if names else f"{i + 1:02d}"
        buf = io.BytesIO()
        cell.save(buf, "PNG")
        out.append((name, buf.getvalue()))
    return out


def _main() -> int:
    """CLI:切分网格图并落盘,输出统一 ``Response`` 形状 JSON。"""
    import argparse
    import sys
    from pathlib import Path

    from windup_common.result import Response

    p = argparse.ArgumentParser(description="把网格图切分为独立 PNG")
    p.add_argument("input", help="输入网格图路径")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--grid", default="2x2", help="网格布局 列x行,如 2x2 / 3x3 / 2x4(默认 2x2)")
    p.add_argument("--names", default=None, help="逗号分隔的文件名(不含 .png);默认 01,02,...")
    args = p.parse_args()

    try:
        cols, rows = (int(x) for x in args.grid.lower().split("x"))
    except ValueError:
        print(Response.fail(f"--grid 格式应为 列x行,收到 {args.grid!r}", code=400).model_dump_json())
        return 2

    names = [n.strip() for n in args.names.split(",")] if args.names else None
    try:
        cells = slice_grid_bytes(Path(args.input).read_bytes(), cols, rows, names)
    except (ValueError, OSError) as exc:
        print(Response.fail(str(exc), code=400).model_dump_json())
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in cells:
        path = out_dir / f"{name}.png"
        path.write_bytes(data)
        paths.append(str(path))

    print(Response.success({"cells": len(cells), "paths": paths}).model_dump_json())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
