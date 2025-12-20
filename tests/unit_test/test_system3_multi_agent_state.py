from internnav.agent.system3.multi_agent.schemas import Milestone, MilestoneProgress
from internnav.agent.system3.multi_agent.state import MultiAgentSystem3State


def test_multi_agent_state_advance_if_done_moves_cursor():
    s = MultiAgentSystem3State(user_goal="g")
    s.milestones = [
        Milestone(id="m1", title="a", completion_criteria=""),
        Milestone(id="m2", title="b", completion_criteria=""),
    ]
    s.current_milestone_idx = 0
    s.progress = [MilestoneProgress(milestone_id="m1", state="DONE", evidence="x")]
    s.advance_if_done()
    assert s.current_milestone_idx == 1


def test_multi_agent_state_get_current_milestone_bounds():
    s = MultiAgentSystem3State(user_goal="g")
    assert s.get_current_milestone() is None
    s.milestones = [Milestone(id="m1", title="a", completion_criteria="")]
    s.current_milestone_idx = 0
    assert s.get_current_milestone().id == "m1"
    s.current_milestone_idx = 3
    assert s.get_current_milestone() is None

from internnav.agent.system3.multi_agent.schemas import Milestone, MilestoneProgress
from internnav.agent.system3.multi_agent.state import MultiAgentSystem3State


def test_multi_agent_state_advances_on_done():
    st = MultiAgentSystem3State(user_goal="g", current_instruction="g")
    st.milestones = [
        Milestone(id="m1", title="first", completion_criteria="x"),
        Milestone(id="m2", title="second", completion_criteria="y"),
    ]
    st.progress = [
        MilestoneProgress(milestone_id="m1", state="DONE", evidence="ok"),
        MilestoneProgress(milestone_id="m2", state="PENDING", evidence=""),
    ]
    assert st.get_current_milestone().id == "m1"
    st.advance_if_done()
    assert st.get_current_milestone().id == "m2"




