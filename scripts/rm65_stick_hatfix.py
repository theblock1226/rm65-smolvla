#!/usr/bin/env python3
"""兼容入口：逻辑已合入 rm65_stick.py。"""

from rm65_stick import *  # noqa: F403
from rm65_stick import StickIntegrator, _axis_ids, _env_bool, _make_stick

__all__ = ["StickIntegrator", "_axis_ids", "_env_bool", "_make_stick"]
