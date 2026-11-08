import json

import pytest

from automl_agent.tools import ToolRegistry, parse_response_calls, validate_candidate_code
from automl_agent.strategy import DECISION_SCHEMA


def _model_experiment_registry(max_items=2):
    calls = []
    registry = ToolRegistry()
    experiment = {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "base_trial_id": {"type": "string"},
            "parameters": {
                "type": "object",
                "properties": {"depth": {"type": "integer"}},
                "maxProperties": 16,
                "additionalProperties": False,
            },
            "decision": DECISION_SCHEMA,
        },
        "required": ["hypothesis", "base_trial_id"],
        "additionalProperties": False,
    }
    registry.register(
        "run_model_experiments",
        "Fit recipe variants",
        {
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_items,
                    "items": experiment,
                }
            },
            "required": ["experiments"],
            "additionalProperties": False,
        },
        lambda experiments: calls.extend(experiments) or experiments,
    )
    return registry, calls


def test_model_experiments_decode_schema_containers_and_move_flat_decision_fields():
    registry, calls = _model_experiment_registry()
    rationale = "Compare the measured model with a small depth change."
    experiment = {
        "hypothesis": "Try a smaller tree depth",
        "base_trial_id": "trial_0001",
        "parameters": json.dumps({"depth": 4}),
        "model_family": "CatBoost",
        "observations": ["The baseline used native categories."],
        "previous_result_assessment": "The measured baseline is stable.",
        "rationale": rationale,
        "decision": {"rationale": rationale},
    }

    result = registry.execute("run_model_experiments", {"experiments": json.dumps([experiment])})

    assert result["ok"]
    assert len(calls) == 1
    assert calls[0]["parameters"] == {"depth": 4}
    assert calls[0]["decision"] == {
        "model_family": "CatBoost",
        "observations": ["The baseline used native categories."],
        "previous_result_assessment": "The measured baseline is stable.",
        "rationale": rationale,
    }
    assert not ({"model_family", "observations", "previous_result_assessment", "rationale"} & calls[0].keys())


def test_container_decoding_does_not_coerce_scalar_values():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "parameters": json.dumps({"depth": "4"}),
    }

    result = registry.execute("run_model_experiments", {"experiments": json.dumps([experiment])})

    assert not result["ok"]
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert "expected=integer" in result["error"]["message"]
    assert calls == []


def test_container_decoding_does_not_allow_forbidden_parameters():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "parameters": json.dumps({"depth": 4, "auto_class_weights": "Balanced"}),
    }

    result = registry.execute("run_model_experiments", {"experiments": json.dumps([experiment])})

    assert not result["ok"]
    assert "auto_class_weights" in result["error"]["message"]
    assert "no additional properties" in result["error"]["message"]
    assert calls == []


def test_flat_decision_conflicts_are_rejected_explicitly():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "rationale": "flat version",
        "decision": {"rationale": "nested version"},
    }

    result = registry.execute("run_model_experiments", {"experiments": json.dumps([experiment])})

    assert not result["ok"]
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert "conflict" in result["error"]["message"].lower()
    assert "rationale" in result["error"]["message"]
    assert calls == []


def test_decision_string_list_metadata_is_wrapped_without_changing_experiments():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "parameters": {"depth": 4},
        "decision": {
            "observations": "The feature has many missing values.",
            "risks": "The signal may not generalize.",
            "alternatives": "Leave preprocessing unchanged.",
        },
    }

    result = registry.execute("run_model_experiments", {"experiments": [experiment]})

    assert result["ok"]
    assert result["value"][0]["decision"] == {
        "observations": ["The feature has many missing values."],
        "risks": ["The signal may not generalize."],
        "alternatives": ["Leave preprocessing unchanged."],
    }
    assert result["value"][0]["parameters"] == {"depth": 4}
    assert calls == result["value"]


def test_decision_empty_string_is_not_fabricated():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "decision": {"observations": ""},
    }

    result = registry.execute("run_model_experiments", {"experiments": [experiment]})

    assert not result["ok"]
    assert "expected=array" in result["error"]["message"]
    assert calls == []


def test_decision_list_normalization_does_not_coerce_numeric_parameters():
    registry, calls = _model_experiment_registry()
    experiment = {
        "hypothesis": "Try a parameter",
        "base_trial_id": "trial_0001",
        "parameters": {"depth": "4"},
        "decision": {"observations": "Metadata only"},
    }

    result = registry.execute("run_model_experiments", {"experiments": [experiment]})

    assert not result["ok"]
    assert "expected=integer" in result["error"]["message"]
    assert calls == []


def test_invalid_large_container_error_is_compact_and_does_not_echo_payload():
    registry, _ = _model_experiment_registry()
    marker = "private-payload-marker"
    result = registry.execute("run_model_experiments", {"experiments": marker * 10000})

    assert not result["ok"]
    message = result["error"]["message"]
    assert len(message) < 300
    assert "$.experiments" in message
    assert "expected=array" in message
    assert marker not in message


def test_decoded_arrays_still_obey_required_fields_and_max_items():
    registry, calls = _model_experiment_registry(max_items=1)
    valid = {"hypothesis": "One", "base_trial_id": "trial_0001"}
    missing_required = {"hypothesis": "Missing base id"}
    too_many = [valid, {"hypothesis": "Two", "base_trial_id": "trial_0002"}]

    missing = registry.execute("run_model_experiments", {"experiments": json.dumps([missing_required])})
    overflow = registry.execute("run_model_experiments", {"experiments": json.dumps(too_many)})

    assert not missing["ok"] and "base_trial_id" in missing["error"]["message"]
    assert not overflow["ok"] and "maxItems=1" in overflow["error"]["message"]
    assert calls == []


def test_native_model_experiment_arrays_remain_accepted():
    registry, calls = _model_experiment_registry()
    experiment = {"hypothesis": "Native array", "base_trial_id": "trial_0001", "parameters": {"depth": 3}}

    result = registry.execute("run_model_experiments", {"experiments": [experiment]})

    assert result["ok"]
    assert result["value"] == [experiment]
    assert calls == [experiment]


@pytest.mark.parametrize("suffix", ["}", "}}"])
def test_encoded_container_with_up_to_two_stray_closing_braces_is_repaired(suffix):
    registry, calls = _model_experiment_registry()
    experiment = {"hypothesis": "One", "base_trial_id": "trial_0001"}
    encoded = json.dumps([experiment]) + suffix

    result = registry.execute("run_model_experiments", {"experiments": encoded})

    assert result["ok"]
    assert result["value"] == [experiment]
    assert calls == [experiment]


@pytest.mark.parametrize("suffix", [" {\"unsafe\":1}", "garbage", "]}}}"])
def test_encoded_container_rejects_nonclosing_or_excess_trailing_data(suffix):
    registry, calls = _model_experiment_registry()
    encoded = json.dumps([{"hypothesis": "One", "base_trial_id": "trial_0001"}]) + suffix

    result = registry.execute("run_model_experiments", {"experiments": encoded})

    assert not result["ok"]
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert "expected=array" in result["error"]["message"]
    assert calls == []


def test_tool_schema_is_enforced_before_side_effect():
    calls = []
    registry = ToolRegistry()
    registry.register("fit", "Fit", {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"], "additionalProperties": False}, lambda n: calls.append(n) or n)
    assert registry.execute("fit", '{"n": 3}')["value"] == 3
    assert not registry.execute("fit", '{"n": -1}')["ok"]
    assert not registry.execute("fit", '{broken')["ok"]
    assert not registry.execute("unknown", '{}')["ok"]
    assert calls == [3]


def test_native_and_text_json_calls_preserve_arguments():
    args = {"hypothesis": "Try a tree", "code": "def build_model(context):\n    return None\n"}
    native = {"tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "run_experiment", "arguments": json.dumps(args)}}]}
    assert parse_response_calls(native)[0]["id"] == "call-1"
    text = {"content": '<think>hidden</think>```json\n' + json.dumps({"tool": "run_experiment", "arguments": args}) + '\n```'}
    call = parse_response_calls(text)[0]
    assert call["function"]["name"] == "run_experiment"
    assert json.loads(call["function"]["arguments"]) == args


def test_generated_code_contract_accepts_model_pipeline():
    validate_candidate_code("from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n")


@pytest.mark.parametrize("code", [
    "def broken(:", "x = 1", "import os\ndef build_model(context):\n return os.system('true')",
    "def build_model(context):\n return open('/tmp/x')", "def build_model(context):\n return object.__subclasses__()",
])
def test_generated_code_obvious_unsafe_or_invalid_operations_rejected(code):
    with pytest.raises(ValueError):
        validate_candidate_code(code)


@pytest.mark.parametrize('source', [
    "import pandas as pd\ndef build_model(context):\n return pd.read_table('/tmp/validation_target.csv', sep=',')",
    "from pandas import read_table as reader\ndef build_model(context):\n return reader('/tmp/validation_target.csv')",
    "def build_model(context):\n return __file__.rsplit('/', 2)[0]",
])
def test_candidate_cannot_use_alternative_file_readers_or_worker_file_location(source):
    with pytest.raises(ValueError):
        validate_candidate_code(source)
