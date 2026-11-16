"""The scorer (s1eval.scoring, s1eval.equivalence). Synthetic values only."""
import json
import shutil
from pathlib import Path

import pytest

from s1eval import equivalence as E
from s1eval.scoring import argument_verdicts, same

DATA = Path(__file__).resolve().parents[2] / "data" / "catalogues"
CT, COUNT, ELIG = "clinicaltrials_search_studies", "clinicaltrials_get_study_count", "clinicaltrials_find_eligible"
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="Node.js is needed for the ClinicalTrials.gov rules")

# ---- the main verdict and underspecified requests ------------------------------------------------------------------
PROPS = {"data_dir": {"type": "string"}, "num_classes": {"type": "integer"}, "output_dir": {"type": "string"}}
GOLD = {"num_classes": 3}


def correct(args, missing, ask=()):
    return argument_verdicts("t", PROPS, GOLD, args, [], missing, ask)["call_correct"]


def test_asking_for_the_missing_value_is_correct():
    assert correct({"num_classes": 3}, ["data_dir"], ["data_dir"])


def test_inventing_the_missing_value_fails():
    assert not correct({"num_classes": 3, "data_dir": "/data/x"}, [], ["data_dir"])


def test_without_ask_a_missing_required_value_still_fails():
    assert not correct({"num_classes": 3}, ["data_dir"])
    assert correct({"num_classes": 3, "data_dir": "/data/x"}, [])


def test_a_wrong_stated_value_fails_even_when_asking():
    assert not correct({"num_classes": 2}, ["data_dir"], ["data_dir"])


def test_normalisation():
    assert same("0", 0) and same("Liver, Spleen", ["spleen", "liver"]) and same(2, 2.0) and not same("a", "b")
    assert same({"PatientID": "P1"}, {"patientid": "p1"})


# ---- imaging servers: name tables and auto-detection ----------------------------------------------------------------
def test_method_names_follow_the_server_table():
    assert E.equivalent("feature_importance_analysis", "method", "random forest", "rf")
    assert E.equivalent("feature_importance_analysis", "method", "Random-Forest", "rf")
    assert E.equivalent("feature_importance_analysis", "method", "ANOVA", "f_test")
    assert not E.equivalent("feature_importance_analysis", "method", "mutual information", "rf")


def test_list_items_are_mapped_then_compared_as_sets():
    assert E.equivalent("pycaret_classification", "include_models", "random forest, extreme gradient boosting", "rf, xgboost")
    assert not E.equivalent("pycaret_classification", "include_models", ["rf"], "rf, xgboost")
    assert E.equivalent("totalsegmentator_segment", "roi_subset", "kidney_left, kidney_right, liver", "liver, kidneys")
    assert E.canonical("totalsegmentator_segment", "roi_subset", "kidneys") == {"kidney_left", "kidney_right"}
    assert E.equivalent("pycaret_classification", "exclude_models", "lightgbm classifier", "lightgbm")
    assert not E.equivalent("pycaret_regression", "exclude_models", "lightgbm regressor", "lightgbm")   # no table there


def test_radiomics_filters_use_the_servers_filter_table():
    assert E.equivalent("pyradiomics_extract_features", "image_types", ["SquareRoot", "Exponential"],
                        ["square root", "exponential"])
    assert E.equivalent("pyradiomics_extract_features", "image_types", "LoG, sqrt", ["laplacian of gaussian", "square root"])
    assert not E.equivalent("pyradiomics_extract_features", "image_types", ["LBP3D"], ["lbp"])       # 'lbp' is LBP2D


def test_no_table_means_no_equivalence():
    assert E.canonical("totalsegmentator_segment", "device", "cpu") is None
    assert not E.equivalent("totalsegmentator_segment", "device", "CPU", "cpu")


def test_autodetection_uses_the_servers_name_lists():
    assert E.autodetected("pycaret_classification", "target_column", "Outcome")
    assert E.autodetected("pycaret_classification_inference", "ground_truth_column", "label")
    assert not E.autodetected("pycaret_regression", "target_column", "price")
    assert not E.autodetected("enhanced_eda_analyze", "target_column", "outcome")


def test_verdicts_keep_strict_and_conditional_apart():
    v = argument_verdicts("pycaret_classification", {}, {"input_path": "/d.csv", "target_column": "Outcome"},
                          {"input_path": "/d.csv"}, [], [])
    assert v["args_correct"] == {"input_path": True, "target_column": False}
    assert v["args_conditional"] == ["target_column"]
    assert not v["call_exact"] and not v["call_exact_equivalent"] and v["call_exact_conditional"]


# ---- biomedical servers --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (DATA / "biomedical_tools.json").exists(), reason="the evaluation set is not next to the code")
def test_every_rule_names_an_argument_of_the_catalogue():
    tools = {t["name"]: t for t in json.loads((DATA / "biomedical_tools.json").read_text())["tools"]}
    for tool, arg in set(E.SERVER_FUNCTIONS) | set(E.JS_TABLES) | E.SEARCH_TEXT:
        assert arg in tools[tool]["input_schema"]["properties"], (tool, arg)


@needs_node
@pytest.mark.parametrize("tool", [CT, COUNT])
def test_status_and_phase_follow_the_server(tool):
    assert E.equivalent(tool, "statusFilter", "Active, not recruiting", "ACTIVE_NOT_RECRUITING")
    assert E.equivalent(tool, "statusFilter", ["recruiting", "not yet recruiting"], "RECRUITING|NOT_YET_RECRUITING")
    assert not E.equivalent(tool, "statusFilter", "recruiting", "COMPLETED")
    assert E.equivalent(tool, "phaseFilter", "Phase 3", "PHASE3")
    assert E.equivalent(tool, "phaseFilter", "Not Applicable", "NA")
    assert not E.equivalent(tool, "phaseFilter", "PHASE2", ["PHASE2", "PHASE3"])


@needs_node
def test_location_synonyms_follow_the_server():
    assert E.equivalent(ELIG, "location", {"country": "UK", "city": "Leeds"}, {"country": "United Kingdom", "city": "leeds"})
    assert E.equivalent(ELIG, "location", {"country": "USA", "state": "MA"},
                        {"country": "United States", "state": "Massachusetts"})
    assert not E.equivalent(ELIG, "location", {"country": "France"}, {"country": "United Kingdom"})


def test_assay_type_table():
    assert E.equivalent("chembl_get_bioactivities", "assay_type", "binding", "B")
    assert E.equivalent("chembl_get_bioactivities", "assay_type", " Functional ", "F")
    assert not E.equivalent("chembl_get_bioactivities", "assay_type", "binding", "F")


def test_search_text_content_words(monkeypatch):
    assert E.content_words("The treatment of asthma") == {"treatment", "asthma"}
    assert E.equivalent(CT, "conditionQuery", "non-small cell lung cancer", "lung cancer, non-small cell")
    assert E.equivalent(ELIG, "conditions", ["Type 2 diabetes", "obesity"], ["Obesity", "diabetes type 2"])
    assert not E.equivalent(CT, "conditionQuery", "breast cancer", "breast neoplasm")
    monkeypatch.setattr(E, "SEARCH_TEXT_RULE", False)             # the strict sensitivity variant
    assert not E.equivalent(CT, "conditionQuery", "non-small cell lung cancer", "lung cancer, non-small cell")


def test_pycaret_sensitivity_table(monkeypatch):
    monkeypatch.setattr(E, "equivalent", E.equivalent)            # restored after the test
    assert not E.equivalent("pycaret_regression", "include_models", "elastic net", "en")
    E.use_pycaret_lists()
    assert E.equivalent("pycaret_regression", "include_models", "elastic net, random forest", "en, rf")
    assert E.equivalent("pycaret_regression", "exclude_models", "Random Forest Regressor", "rf")
