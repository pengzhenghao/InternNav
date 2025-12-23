"""
Agent package entrypoints.

Important: keep `internnav.agent` import-light.

Some agents depend on heavy optional stacks (Habitat, Gym, large model weights).
Importing them eagerly breaks lightweight use-cases (e.g., Sys3-only orchestration
inside other simulators like SimWorld). We therefore expose agents via **lazy imports**.
"""

from __future__ import annotations

from internnav.agent.base import Agent

__all__ = [
    'Agent',
    'DialogAgent',
    'CmaAgent',
    'RdpAgent',
    'Seq2SeqAgent',
    'InternVLAN1Agent',
    'System3Agent',
    'Sys3OnlyAgent',
    'Sys3OnlyAgentCfg',
]


def __getattr__(name: str):
    # Lazy imports to avoid importing optional dependencies at package import time.
    if name == "CmaAgent":
        from internnav.agent.cma_agent import CmaAgent

        return CmaAgent
    if name == "RdpAgent":
        from internnav.agent.rdp_agent import RdpAgent

        return RdpAgent
    if name == "Seq2SeqAgent":
        from internnav.agent.seq2seq_agent import Seq2SeqAgent

        return Seq2SeqAgent
    if name == "InternVLAN1Agent":
        from internnav.agent.internvla_n1_agent import InternVLAN1Agent

        return InternVLAN1Agent
    if name == "System3Agent":
        from internnav.agent.system3_agent import System3Agent

        return System3Agent
    if name == "Sys3OnlyAgent":
        from internnav.agent.sys3_only_agent import Sys3OnlyAgent

        return Sys3OnlyAgent
    if name == "Sys3OnlyAgentCfg":
        from internnav.agent.sys3_only_agent import Sys3OnlyAgentCfg

        return Sys3OnlyAgentCfg

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
