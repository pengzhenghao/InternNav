from __future__ import annotations

"""
External state store for multi-agent System 3.

Extends the single-agent `System3State` with explicit milestone memory:
- milestone list (static for the episode)
- progress records (state + evidence)
- current milestone pointer
"""

from dataclasses import dataclass, field
from typing import List, Optional

from internnav.agent.system3.state import System3State

from .schemas import Milestone, MilestoneProgress


@dataclass
class MultiAgentSystem3State(System3State):
    milestones: List[Milestone] = field(default_factory=list)
    progress: List[MilestoneProgress] = field(default_factory=list)
    current_milestone_idx: int = 0

    def get_current_milestone(self) -> Optional[Milestone]:
        if 0 <= self.current_milestone_idx < len(self.milestones):
            return self.milestones[self.current_milestone_idx]
        return None

    def get_progress(self, milestone_id: str) -> Optional[MilestoneProgress]:
        for p in self.progress:
            if p.milestone_id == milestone_id:
                return p
        return None

    def upsert_progress(self, update: MilestoneProgress) -> None:
        for i, p in enumerate(self.progress):
            if p.milestone_id == update.milestone_id:
                self.progress[i] = update
                return
        self.progress.append(update)

    def advance_if_done(self) -> None:
        cur = self.get_current_milestone()
        if not cur:
            return
        p = self.get_progress(cur.id)
        if p and p.state == "DONE":
            self.current_milestone_idx = min(self.current_milestone_idx + 1, max(len(self.milestones) - 1, 0))




