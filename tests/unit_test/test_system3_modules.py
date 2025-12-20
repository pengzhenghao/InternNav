import os
import pytest

from internnav.agent.system3.compiler import InstructionCompiler
from internnav.agent.system3.critic import System3Critic
from internnav.agent.system3.schemas import System3Plan, System3PromptProfile
from internnav.agent.system3.state import System3State


def _dbg_enabled() -> bool:
    # Pytest captures stdout by default. To see prints:
    #   pytest -s tests/unit_test/test_system3_modules.py
    # or:
    #   SYSTEM3_TEST_DEBUG=1 pytest -q tests/unit_test/test_system3_modules.py
    return os.environ.get("SYSTEM3_TEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}


def _dbg(title: str, obj) -> None:
    if _dbg_enabled():
        print(f"\n[{title}]")
        print(obj)


def test_plan_parsing_normalizes_status_and_change_flag():
    plan = System3Plan.from_tool_json(
        {
            "thought": "x",
            "status": "nAvIgAtInG",  # invalid -> should become SEARCH
            "instruction": "Walk forward",
            "change_instruction": "yes",
            "discrete_actions": "MOVE_FORWARD",  # invalid type -> []
        }
    )
    _dbg("parsed_plan", plan)
    assert plan.status == "SEARCH"
    assert plan.change_instruction is True
    assert plan.discrete_actions == []


def test_compiler_rejects_banned_phrases_and_forces_search():
    profile = System3PromptProfile(
        name="test",
        system_prompt="",
        instruction_banlist=["on your left", "keep"],
        max_instruction_chars=200,
        max_sentences=2,
    )
    compiler = InstructionCompiler(profile=profile)
    plan = System3Plan(thought="t", status="NAVIGATE", instruction="Keep the wall on your left and proceed.", change_instruction=False)
    _dbg("compiler_input", plan)
    out = compiler.compile(plan)
    _dbg("compiler_output", out)
    # Compiler is minimal formatter now: it should not rewrite intent.
    assert out.status == "NAVIGATE"
    assert out.instruction.lower().startswith("keep the wall on your left")


def test_compiler_adds_punctuation_and_limits_sentences():
    profile = System3PromptProfile(name="test", system_prompt="", instruction_banlist=[], max_instruction_chars=500, max_sentences=2)
    compiler = InstructionCompiler(profile=profile)
    plan = System3Plan(
        thought="t",
        status="NAVIGATE",
        instruction="Turn left. Walk forward. Stop near the chair",
        change_instruction=False,
    )
    _dbg("compiler_input", plan)
    out = compiler.compile(plan)
    _dbg("compiler_output", out)
    # Ends with punctuation
    assert out.instruction[-1] in ".!?"


def test_critic_forces_recovery_when_no_visual_change():
    profile = System3PromptProfile(name="test", system_prompt="")
    critic = System3Critic(profile=profile, stuck_no_change_threshold=2)
    state = System3State(user_goal="g")
    state.no_change_count = 2
    plan = System3Plan(thought="t", status="NAVIGATE", instruction="Walk forward.", change_instruction=False)
    _dbg("critic_state", state)
    _dbg("critic_input", plan)
    out = critic.post_process(state, plan)
    _dbg("critic_output", out)
    # Critic is no-op now: it should not rewrite the plan.
    assert out.status == "NAVIGATE"
    assert out.instruction.startswith("Walk forward")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
