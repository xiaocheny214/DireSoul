"""奔跑 i2v 提示词(视频路线,循环类)。

与 walk 同机制、更强:前倾、步幅更大、每步有**腾空相**(双脚离地)、衣摆后扬。
要点同 walk(Issue #35):只写正向词、逐条写腿部可见动作、锁死手持武器、朝向必须与母版一致。
换角色只替换 garment / feet 装备子句,机制词不动。
"""

from __future__ import annotations

__all__ = ["RUN_BODY_SIDE", "RUN_BODY_FRONT", "DEFAULT_GARMENT", "build_run_prompt"]

# 侧跑(横版):快速向右推进 + 腾空相 + 前倾。
RUN_BODY_SIDE = (
    "The character runs fast to the right through the open space, the whole body driving "
    "forward with each long stride: the front boot reaches far forward and plants, the rear "
    "boot pushes off hard so both boots leave the ground at the peak of each stride, the "
    "torso leans forward into the run, the free arm pumps with the rhythm, {garment} stream "
    "backward behind the body from the speed, the weapon stays held firmly at the side in a "
    "fixed grip, SIDE VIEW facing right the whole time, the legs clearly visible with a long "
    "reaching stride."
)

# 正面跑(俯视 / 2.5D):朝观者原地奔跑,高抬腿,不转身。
RUN_BODY_FRONT = (
    "The character runs in place toward the viewer, sprinting on the spot: each knee drives "
    "up high toward the camera in turn while the other boot pushes off hard, the torso leans "
    "forward with the effort, {garment} stream backward behind the body, the weapon stays "
    "held firmly in a fixed grip, the character keeps FACING THE VIEWER and stays centered in "
    "frame, both legs clearly visible with a fast high-knee rhythm."
)

DEFAULT_GARMENT = "the cape and tabard"


def build_run_prompt(
    garment: str = DEFAULT_GARMENT, feet: str = "boot", facing: str = "side"
) -> str:
    """按角色装备 + 母版朝向生成奔跑正文。``facing`` 须与母版朝向一致(side / front)。"""
    if facing not in ("side", "front"):
        raise ValueError(f"facing 只能是 'side' 或 'front',收到 {facing!r}")
    template = RUN_BODY_SIDE if facing == "side" else RUN_BODY_FRONT
    body = template.format(garment=garment)
    if feet != "boot":
        body = body.replace("boot", feet)
    return body
