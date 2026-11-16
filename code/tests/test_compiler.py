"""The compiler (s1agent.compiler), with scripted backends standing in for the decision model."""
from s1agent.backends.base import Backend, Decision
from s1agent.catalog import ToolSpec
from s1agent.compiler import (NO_ITEM, NOT_SPECIFIED, ON, UNMENTIONED, ArgumentFiller, documented_default, flat_schema,
                              is_default, param_kind)


class Scripted(Backend):
    """Answers from a {argument name: option} script; unscripted questions get the 'keep default' option, yes/no gets no."""
    name = "scripted"

    def __init__(self, script, p=0.9):
        super().__init__()
        self.script, self.p = script, p

    def _choice(self, state, question, options, kind):
        want = next((v for k, v in self.script.items() if f"`{k}`" in question), None)
        if kind == "noul":
            want = "yes" if want else "no"
        if want not in options:
            want = next((o for o in options if o in (NOT_SPECIFIED, UNMENTIONED)), options[-1])
        return Decision(question, options, [self.p if o == want else (1 - self.p) / (len(options) - 1) for o in options],
                        kind)


TOOL = ToolSpec("eda", "enhanced_eda_analyze", "Exploratory data analysis.", {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input data file"},
        "output_dir": {"type": "string", "description": "Output directory"},
        "target_column": {"type": "string", "description": "Target column"},
        "correlation_method": {"type": "string", "enum": ["pearson", "spearman"]},
        "dpi": {"type": "integer", "description": "Figure resolution"},
        "vif_analysis": {"type": "boolean", "description": "Calculate VIF"},
        "ignore": {"type": "string", "description": "Comma-separated columns to ignore"},
    },
    "required": ["input_path", "output_dir"],
})


def test_param_kinds():
    p = TOOL.input_schema["properties"]
    assert [param_kind(n, p[n]) for n in p] == ["path", "path", "column", "enum", "number", "bool", "list"]
    assert param_kind("filters", {"type": "object"}) == "object"


def test_question_texts_are_fixed():
    """The question and option texts are hashed into the answer-store keys: they must never change."""
    f = ArgumentFiller(Scripted({}))
    qs = {name: spec for name, spec, _ in f.questions("EDA of /d.csv with 600 dpi, ignore pid", TOOL)}
    assert qs["input_path"]["state"] == "User request: EDA of /d.csv with 600 dpi, ignore pid\nTool: enhanced_eda_analyze: " \
                                        "Exploratory data analysis."
    assert qs["input_path"]["question"] == "Which value should the argument `input_path` take? `input_path`: Path to input " \
                                           "data file"
    assert qs["input_path"]["options"] == ["/d.csv", "not specified in the request (keep the default)"]
    assert qs["vif_analysis"]["question"] == "Does the request turn the option `vif_analysis` on or off? `vif_analysis`: " \
                                             "Calculate VIF"
    assert qs["vif_analysis"]["options"] == ["yes: turn it on", "no: turn it off",
                                             "not mentioned in the request (keep the default)"]
    assert qs["correlation_method"]["options"] == ["pearson", "spearman", NOT_SPECIFIED]
    assert qs["ignore"]["question"] == "Which item does the request want in `ignore`? `ignore`: Comma-separated columns to " \
                                       "ignore"
    assert qs["ignore"]["options"][-1] == "none: the request names no item for it"
    assert qs["dpi"]["options"] == ["600", NOT_SPECIFIED]


def test_fill_sets_only_what_the_model_picks(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("pid,age,outcome\n")
    req = f"EDA of {csv} with spearman, target outcome, 600 dpi, VIF on"
    b = Scripted({"input_path": str(csv), "target_column": "outcome", "correlation_method": "spearman", "dpi": "600",
                  "vif_analysis": ON})
    f = ArgumentFiller(b, {"output_dir": "{workspace}/{tool}"}, workspace="/ws").fill(req, TOOL)
    assert f.args == {"input_path": str(csv), "output_dir": "/ws/enhanced_eda_analyze", "target_column": "outcome",
                      "correlation_method": "spearman", "dpi": 600, "vif_analysis": True}
    assert isinstance(f.args["dpi"], int)
    assert not f.missing and not f.errors


def test_independent_questions_go_out_in_one_batch():
    rounds = []

    class Recorder(Scripted):
        def choice_batch(self, specs):
            rounds.append(sorted(s["tag"].split(".")[-1] for s in specs))
            return super().choice_batch(specs)

    b = Recorder({"input_path": "/d.csv", "correlation_method": "spearman"})
    f = ArgumentFiller(b, {"output_dir": "/o"}).fill("EDA of /d.csv with spearman, ignore pid", TOOL)
    assert rounds == [["correlation_method", "ignore:item", "input_path", "output_dir", "target_column", "vif_analysis"]]
    assert f.args == {"input_path": "/d.csv", "output_dir": "/o", "correlation_method": "spearman"}


def test_required_without_candidates_is_missing():
    f = ArgumentFiller(Scripted({})).fill("run an EDA please", TOOL)
    assert set(f.missing) == {"input_path", "output_dir"}
    assert f.args == {}


def test_documented_default():
    assert documented_default({"default": 10}) == 10
    assert documented_default({"description": "Which model to use. Default: 'total' for CT."}) == "total"
    assert documented_default({"description": "Use colorblind-safe palettes (default: true)"}) is True
    assert documented_default({"description": "Spacing. Default: [1, 1, 1]"}) == [1, 1, 1]
    assert documented_default({"description": "Target column name."}) is None
    assert is_default("1.0", {"default": 1}) and not is_default(True, {"default": 1})


def test_a_value_fills_one_argument_and_defaults_are_not_passed():
    tool = ToolSpec("ts", "seg", "Segment.", {"type": "object", "required": ["input_path"], "properties": {
        "input_path": {"type": "string", "description": "Path to the image"},
        "crop_path": {"type": "string", "description": "Custom path to masks used for cropping"},
        "output_type": {"type": "string", "enum": ["nifti", "dicom"], "description": "Format (default: nifti)"}}})
    class Owner(Scripted):
        def _choice(self, state, question, options, kind):
            if question.startswith("The request gives the value"):      # both want the path: it is input_path's
                return Decision(question, options, [0.9 if o.startswith("`input_path`") else 0.1 for o in options], kind)
            return super()._choice(state, question, options, kind)
    b = Owner({"input_path": "/d/ct.nii.gz", "crop_path": "/d/ct.nii.gz", "output_type": "nifti"})
    f = ArgumentFiller(b).fill("Segment /d/ct.nii.gz as nifti", tool)
    assert f.args == {"input_path": "/d/ct.nii.gz"}                 # the path is written once: one argument takes it
    assert {d.name: d.status for d in f.decisions}["output_type"] == "is_default"


def test_a_value_written_twice_can_fill_two_arguments():
    tool = ToolSpec("x", "t", "Train.", {"type": "object", "properties": {
        "num_classes": {"type": "integer", "description": "Number of classes"},
        "gamma": {"type": "number", "description": "Focal loss gamma"}}})
    f = ArgumentFiller(Scripted({"num_classes": "2", "gamma": "2"})).fill("2 classes, focal gamma 2", tool)
    assert f.args == {"num_classes": 2, "gamma": 2.0}


def test_optional_needs_confidence_that_it_is_specified():
    class Unsure(Scripted):
        """Top value 0.5, 'not specified' 0.4: the value wins but P(specified) = 0.6 < 0.7."""
        def _choice(self, state, question, options, kind):
            d = super()._choice(state, question, options, kind)
            top = max(range(len(options)), key=d.probs.__getitem__)
            ns = next((i for i, o in enumerate(options) if o in (NOT_SPECIFIED, UNMENTIONED)), None)
            if ns is not None and ns != top:
                rest = 0.1 / max(1, len(options) - 2)
                d.probs = [0.5 if i == top else 0.4 if i == ns else rest for i in range(len(options))]
            return d
    f = ArgumentFiller(Unsure({"input_path": "/d.csv", "correlation_method": "spearman"}), {"output_dir": "/o"}
                       ).fill("EDA of /d.csv", TOOL)
    assert "correlation_method" not in f.args
    assert {d.name: d.status for d in f.decisions}["correlation_method"] == "low_conf"


def test_list_items_are_picked_one_at_a_time():
    tool = ToolSpec("pc", "train", "AutoML.", {"type": "object", "properties": {
        "include_models": {"type": "string", "description": "Comma-separated models to include"},
        "exclude_models": {"type": "string", "description": "Comma-separated models to exclude"}}})

    class Picker(Backend):
        """include_models: rf, then xgboost, then stops; exclude_models: none."""
        name = "picker"

        def _choice(self, state, question, options, kind):
            want = options[-1]
            if "`include_models`" in question:
                want = next((o for o in ("rf", "xgboost") if o in options and f" {o}" not in question.split("chosen:")[-1]),
                            options[-1])
            return Decision(question, options, [0.9 if o == want else 0.1 / (len(options) - 1) for o in options], kind)
    b = Picker()
    f = ArgumentFiller(b).fill("train on the tumour data but only include rf and xgboost", tool)
    assert f.args == {"include_models": "rf, xgboost"}
    first = [d for d in b.trace if "`include_models`" in d.question]
    assert first[0].options[-1] == NO_ITEM and first[1].question.endswith("Already chosen: rf.")


def test_an_object_is_a_written_dictionary_or_a_gate_then_fields():
    tool = ToolSpec("x", "t", "Augment.", {"type": "object", "properties": {
        "augment": {"type": "object", "description": "Augmentation settings"}}})
    f = ArgumentFiller(Scripted({"augment": '{"rotation": 20}'})).fill('augment with {"rotation": 20}', tool)
    assert f.args == {"augment": {"rotation": 20}}
    b = Scripted({"augment": True, "PatientID": "P07"})

    def answer(state, question, options, kind):                     # the gate says yes, the field PatientID is P07
        want = "yes" if kind == "noul" else "PatientID" if "Which field" in question and "PatientID" in options \
            else "P07" if "`PatientID`" in question else options[-1]
        return Decision(question, options, [0.9 if o == want else 0.1 / (len(options) - 1) for o in options], kind)
    b._choice = answer
    f = ArgumentFiller(b).fill("only the PatientID P07", tool)
    assert f.args == {"augment": {"PatientID": "P07"}}
    gate = next(d for d in b.trace if d.kind == "noul")
    assert gate.question == "Does the request say which items the argument `augment` should contain? `augment`: " \
                            "Augmentation settings"


def test_a_contested_value_is_given_to_the_argument_the_model_names():
    tool = ToolSpec("x", "t", "Copy.", {"type": "object", "properties": {
        "input_path": {"type": "string", "description": "Path to the input"},
        "output_path": {"type": "string", "description": "Path to the output"}}})

    class Both(Scripted):
        def _choice(self, state, question, options, kind):
            if question.startswith("The request gives the value"):
                return Decision(question, options, [0.1, 0.9], kind)  # it belongs to output_path
            return super()._choice(state, question, options, kind)
    b = Both({"input_path": "/d/a.csv", "output_path": "/d/a.csv"})
    f = ArgumentFiller(b).fill("copy /d/a.csv", tool)
    owner = next(d for d in b.trace if d.question.startswith("The request gives the value"))
    assert owner.question == "The request gives the value '/d/a.csv'. Which argument does it belong to?"
    assert owner.options == ["`input_path`: Path to the input", "`output_path`: Path to the output"]
    assert f.args == {"output_path": "/d/a.csv"}


def test_union_schemas_read_as_one_type():
    p = flat_schema({"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "Count", "default": None})
    assert p["type"] == "integer" and p["description"] == "Count"
    p = flat_schema({"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]})
    assert p["type"] == "array"
