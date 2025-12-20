from __future__ import annotations

"""
System 3 State Store (non-LLM, persistent).

This is the "memory" component of the modular System 3 design.
It tracks:
- current instruction lifecycle (sub-episodes)
- recent observation frames (base64), downsampled to a fixed budget
- simple failure signals (no visual change, repeated instruction)

It is intentionally environment-agnostic: it stores only lightweight signals,
not simulator-specific geometry.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

def _image_sig(img_b64: str) -> str:
    # Cheap-ish signature for "no visual change" detection.
    # We avoid decoding the whole image; base64 tail still changes in practice.
    tail = (img_b64 or "")[-2048:].encode("utf-8", errors="ignore")
    return hashlib.md5(tail).hexdigest()


@dataclass
class System3State:
    user_goal: str

    # External counters (informational; set by caller)
    sys1_steps: int = 0
    sys2_calls: int = 0
    sys3_calls: int = 0

    # Instruction lifecycle
    current_instruction: Optional[str] = None
    last_emitted_instruction: Optional[str] = None

    # Sub-episode memory: frames since current instruction was issued
    subepisode_id: int = 0
    max_subepisode_frames: int = 8
    subepisode_frames_b64: List[str] = field(default_factory=list)
    subepisode_frame_sigs: List[str] = field(default_factory=list)

    # Simple failure counters
    no_change_count: int = 0
    same_instruction_count: int = 0


    def reset_episode(self) -> None:
        self.sys1_steps = 0
        self.sys2_calls = 0
        self.sys3_calls = 0
        self.current_instruction = None
        self.last_emitted_instruction = None
        self.subepisode_id = 0
        self.subepisode_frames_b64 = []
        self.subepisode_frame_sigs = []
        self.no_change_count = 0
        self.same_instruction_count = 0

    def start_new_subepisode(self, latest_img_b64: Optional[str] = None) -> None:
        self.subepisode_id += 1
        self.subepisode_frames_b64 = []
        self.subepisode_frame_sigs = []
        if latest_img_b64:
            self.append_frame(latest_img_b64)

    def append_frame(self, img_b64: str) -> List[str]:
        self.subepisode_frames_b64.append(img_b64)
        self.subepisode_frame_sigs.append(_image_sig(img_b64))

        # Update no-change heuristic
        if len(self.subepisode_frame_sigs) >= 2 and self.subepisode_frame_sigs[-1] == self.subepisode_frame_sigs[-2]:
            self.no_change_count += 1
            if self.no_change_count in (1, 2, 3, 5, 8):
                logger.debug("[Sys3] State: no-change count=%d", self.no_change_count)
        else:
            self.no_change_count = 0

        # Downsample uniformly if needed
        if len(self.subepisode_frames_b64) <= self.max_subepisode_frames:
            return self.subepisode_frames_b64

        total = len(self.subepisode_frames_b64)
        # Uniformly sample indices; include endpoints
        if self.max_subepisode_frames <= 1:
            keep = [total - 1]
        else:
            step = (total - 1) / float(self.max_subepisode_frames - 1)
            keep = [int(round(i * step)) for i in range(self.max_subepisode_frames)]
            keep[-1] = total - 1
            keep[0] = 0

        self.subepisode_frames_b64 = [self.subepisode_frames_b64[i] for i in keep]
        self.subepisode_frame_sigs = [self.subepisode_frame_sigs[i] for i in keep]
        logger.debug("[Sys3] State: downsampled subepisode frames: %d -> %d", total, len(self.subepisode_frames_b64))
        return self.subepisode_frames_b64


