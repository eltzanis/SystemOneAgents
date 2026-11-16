"""Tool selection of the agent loop (s1agent.agent), with scripted and lexical backends."""
from s1agent.agent import NONE_FITS, S1Agent
from s1agent.backends.base import Backend, Decision
from s1agent.backends.lexical import LexicalBackend
from s1agent.catalog import ToolSpec

TOOLS = [ToolSpec("seg", "segment", "Segment organs in a CT image.", {"type": "object", "required": ["input_path"],
                  "properties": {"input_path": {"type": "string", "description": "Path to the CT image"}}}),
         ToolSpec("eda", "analyze", "Exploratory data analysis of a CSV file.", {"type": "object", "properties": {
             "input_path": {"type": "string", "description": "Path to the CSV file"}}})]


class Fixed(Backend):
    """Selection probabilities per tool name; yes/no answers P(yes) = p_yes; argument questions: the first option."""
    name = "fixed"

    def __init__(self, sel, p_yes=0.9, fit=None):
        super().__init__()
        self.sel, self.p_yes, self.fit = sel, p_yes, fit

    def _choice(self, state, question, options, kind):
        if kind == "noul":
            return Decision(question, options, [1 - self.p_yes, self.p_yes], kind)
        if question == "Which tool should be called next for this request?":
            w = [self.sel.get(o.split(" / ")[1].split(":")[0] if " / " in o else "none", 0.0) for o in options]
            return Decision(question, options, w, kind)
        if question.startswith("Which of these two tools"):
            return Decision(question, options, [0.2, 0.8] if self.fit == 1 else [0.8, 0.2], kind)
        return Decision(question, options, [0.9] + [0.1 / (len(options) - 1)] * (len(options) - 1), kind)


def test_selection_question_lists_every_tool_and_none():
    b = Fixed({"segment": 0.9, "analyze": 0.05, "none": 0.05})
    plan = S1Agent(TOOLS, b).plan("Segment /d/ct.nii.gz")
    d = plan.selection.steps[0]
    assert d.question == "Which tool should be called next for this request?"
    assert d.options == ["seg / segment: Segment organs in a CT image.",
                         "eda / analyze: Exploratory data analysis of a CSV file.", NONE_FITS]
    cap = plan.selection.steps[1]
    assert cap.kind == "noul" and cap.question == ("Can the tool `segment` do what the request asks? `segment`: Segment "
                                                   "organs in a CT image. Parameters: input_path.")
    assert plan.selection.tool.name == "segment" and plan.filled.args == {"input_path": "/d/ct.nii.gz"}


def test_none_of_these_abstains():
    plan = S1Agent(TOOLS, Fixed({"none": 0.8, "segment": 0.1, "analyze": 0.1})).plan("What is the weather?")
    assert plan.selection.tool is None and plan.filled is None


def test_close_tools_ask_which_parameters_fit():
    b = Fixed({"segment": 0.45, "analyze": 0.40, "none": 0.15}, fit=1)
    sel = S1Agent(TOOLS, b).select("look at /d/x.csv")
    fit = sel.steps[1]
    assert fit.options == ["segment: parameters input_path", "analyze: parameters input_path"]
    assert sel.tool.name == "analyze"


def test_capability_check_can_abstain():
    sel = S1Agent(TOOLS, Fixed({"segment": 0.9, "analyze": 0.05, "none": 0.05}, p_yes=0.3)).select("Segment it")
    assert sel.tool is None and abs(sel.confidence - 0.7) < 1e-9


def test_lexical_baseline_runs_without_a_model():
    agent = S1Agent(TOOLS, LexicalBackend(), select_min_conf=0.0)
    plan = agent.plan("Segment the organs in the CT image /d/ct.nii.gz")
    assert plan.selection.tool.name == "segment"
