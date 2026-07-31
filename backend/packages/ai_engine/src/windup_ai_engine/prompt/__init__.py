"""prompt:各动作的生成提示词与装配。"""

from .actions import build_attack_prompt, build_idle_prompt
from .jump import JUMP_PHASES, build_jump_prompt
from .run import RUN_BODY_FRONT, RUN_BODY_SIDE, build_run_prompt
from .walk import WALK_BODY_FRONT, WALK_BODY_SIDE, build_walk_prompt

__all__ = [
    "WALK_BODY_SIDE",
    "WALK_BODY_FRONT",
    "build_walk_prompt",
    "RUN_BODY_SIDE",
    "RUN_BODY_FRONT",
    "build_run_prompt",
    "JUMP_PHASES",
    "build_jump_prompt",
    "build_idle_prompt",
    "build_attack_prompt",
]
