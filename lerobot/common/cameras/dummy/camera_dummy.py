# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from typing import Any

import numpy as np

from ..camera import Camera
from .configuration_dummy import DummyCameraConfig


class DummyCamera(Camera):
    """Returns a constant black RGB frame (H, W, 3) uint8."""

    def __init__(self, config: DummyCameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        h = int(config.height or 480)
        w = int(config.width or 640)
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        self._connected = True

    def read(self, color_mode=None) -> np.ndarray:
        return self._frame.copy()

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self._frame.copy()

    def disconnect(self) -> None:
        self._connected = False
