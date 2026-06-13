from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator

import torch


STAGE0_TIMING_COMPONENTS = (
    "prior",
    "pbu",
    "policy",
    "memory",
    "mask_projection",
    "selection",
    "io",
)


@dataclass
class FrameTiming:
    frame_idx: int
    sections_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0


class TimingProfiler:
    def __init__(
        self,
        *,
        sync_cuda: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        self.sync_cuda = bool(sync_cuda)
        self.device = torch.device(device) if device is not None else None
        self.frames: list[FrameTiming] = []
        self._active_frame: FrameTiming | None = None
        self._frame_start_t: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self._active_frame = None
        self._frame_start_t = None

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _now(self) -> float:
        self._sync()
        return time.perf_counter()

    def start_frame(self, frame_idx: int) -> None:
        if self._active_frame is not None:
            raise RuntimeError("Cannot start a new frame while another frame is active")
        self._active_frame = FrameTiming(frame_idx=int(frame_idx))
        self._frame_start_t = self._now()

    def end_frame(self) -> FrameTiming:
        if self._active_frame is None or self._frame_start_t is None:
            raise RuntimeError("No active frame to end")
        frame = self._active_frame
        frame.total_ms = float((self._now() - self._frame_start_t) * 1000.0)
        for name in STAGE0_TIMING_COMPONENTS:
            frame.sections_ms.setdefault(name, 0.0)
        self.frames.append(frame)
        self._active_frame = None
        self._frame_start_t = None
        return frame

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start_t = self._now()
        try:
            yield
        finally:
            elapsed_ms = float((self._now() - start_t) * 1000.0)
            self.record_ms(name, elapsed_ms)

    def record_ms(self, name: str, elapsed_ms: float) -> None:
        if self._active_frame is None:
            return
        previous = float(self._active_frame.sections_ms.get(name, 0.0))
        self._active_frame.sections_ms[name] = previous + float(elapsed_ms)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        return float(sum(values) / len(values))

    def summary(self) -> dict:
        total_ms = [float(frame.total_ms) for frame in self.frames]
        total_s = float(sum(total_ms) / 1000.0)
        steady_ms = total_ms[1:]
        steady_s = float(sum(steady_ms) / 1000.0)
        steady_fps = None
        if len(steady_ms) > 0 and steady_s > 0.0:
            steady_fps = float(len(steady_ms) / steady_s)

        all_names = set(STAGE0_TIMING_COMPONENTS)
        for frame in self.frames:
            all_names.update(frame.sections_ms)
        mean_ms = {
            name: self._mean([float(frame.sections_ms.get(name, 0.0)) for frame in self.frames])
            for name in sorted(all_names)
        }
        mean_excl_first_ms = {
            name: self._mean([float(frame.sections_ms.get(name, 0.0)) for frame in self.frames[1:]])
            for name in sorted(all_names)
        }
        return {
            "n_frames": int(len(self.frames)),
            "total_s": total_s,
            "first_frame_s": None if len(total_ms) == 0 else float(total_ms[0] / 1000.0),
            "steady_fps_excl_first": steady_fps,
            "stage_timing_breakdown_ms": {
                "mean": mean_ms,
                "mean_excl_first": mean_excl_first_ms,
            },
            "frame_metrics": [
                {
                    "frame_idx": int(frame.frame_idx),
                    "total_ms": float(frame.total_ms),
                    **{f"{name}_ms": float(value) for name, value in sorted(frame.sections_ms.items())},
                }
                for frame in self.frames
            ],
        }
