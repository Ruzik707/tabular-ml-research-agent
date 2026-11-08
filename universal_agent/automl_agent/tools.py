"""Explicit capability contracts and conservative candidate linting.

The AST check catches obvious mistakes. It is not a security sandbox; the
process supervisor and the deployment container provide separate controls.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from uuid import uuid4

from jsonschema import Draft202012Validator


_MAX_CONTAINER_DECODE_DEPTH = 8
_MAX_ARGUMENT_ERROR_LENGTH = 300


class _ArgumentNormalizationError(ValueError):
    """A safe-to-report error raised while normalizing a tool call."""

    def __init__(self, path, reason):
        self.path = path
        self.reason = reason
        super().__init__(f"path={path} expected=consistent decision metadata; reason={reason}")


def _schema_types(schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, str):
        return {expected}
    if isinstance(expected, list):
        return {item for item in expected if isinstance(item, str)}
    return set()


def _decode_encoded_container(value, expected):
    """Decode one JSON object/array, allowing at most two stray closing brackets."""
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        start = len(value) - len(value.lstrip())
        try:
            decoded, end = json.JSONDecoder().raw_decode(value, start)
        except (ValueError, TypeError, RecursionError):
            return value
        trailing = value[end:].strip()
        if not 1 <= len(trailing) <= 2 or any(char not in "]}" for char in trailing):
            return value

    if isinstance(decoded, dict) and "object" in expected:
        return decoded
    if isinstance(decoded, list) and "array" in expected:
        return decoded
    return value


def _decode_schema_containers(value, schema, depth=0):
    """Decode JSON strings only at schema locations requiring object/array."""
    if not isinstance(schema, dict) or depth >= _MAX_CONTAINER_DECODE_DEPTH:
        return value

    expected = _schema_types(schema)
    if isinstance(value, str) and expected.intersection({"object", "array"}):
        value = _decode_encoded_container(value, expected)

    if isinstance(value, dict) and "object" in expected:
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties")
        for key, item in list(value.items()):
            child_schema = properties.get(key) if isinstance(properties, dict) else None
            if child_schema is None and isinstance(additional, dict):
                child_schema = additional
            if isinstance(child_schema, dict):
                value[key] = _decode_schema_containers(item, child_schema, depth + 1)
    elif isinstance(value, list) and "array" in expected:
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                value[index] = _decode_schema_containers(item, item_schema, depth + 1)
    return value


def _json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _error_path(parts):
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{json.dumps(str(part), ensure_ascii=False)[:50]}]"
    return path


def _compact_validation_error(error):
    """Format jsonschema failures without rendering user-supplied values."""
    validator = error.validator
    setting = error.validator_value
    instance = error.instance
    expected = f"constraint {validator}"
    reason = "value does not satisfy the constraint"

    if validator == "type":
        expected_type = "|".join(str(item) for item in setting) if isinstance(setting, list) else str(setting)
        expected = expected_type
        reason = f"received {_json_type(instance)}"
    elif validator == "required":
        missing = sorted(set(setting) - set(instance)) if isinstance(instance, dict) else []
        expected = "required fields"
        reason = "missing " + ", ".join(str(item)[:60] for item in missing[:8]) if missing else "required field is absent"
    elif validator == "additionalProperties":
        properties = error.schema.get("properties", {})
        known = set(properties) if isinstance(properties, dict) else set()
        extra = sorted(set(instance) - known) if isinstance(instance, dict) else []
        expected = "no additional properties"
        reason = "unexpected key(s): " + ", ".join(str(item)[:60] for item in extra[:8]) if extra else "unexpected property"
    elif validator in {"minItems", "maxItems", "minProperties", "maxProperties", "minLength", "maxLength"}:
        expected = f"{validator}={setting}"
        size = len(instance) if isinstance(instance, (list, dict, str)) else None
        unit = "items" if isinstance(instance, list) else "properties" if isinstance(instance, dict) else "characters"
        reason = f"received {size} {unit}" if size is not None else "value has the wrong shape"
    elif validator in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}:
        expected = f"{validator}={setting}"
        reason = "numeric value is outside the allowed range"
    elif validator == "pattern":
        expected = "pattern constraint"
        reason = f"string must match {str(setting)[:100]}"
    elif validator == "enum":
        choices = json.dumps(setting, ensure_ascii=False, separators=(",", ":"))[:120]
        expected = f"one of {choices}"
        reason = "value is outside allowed choices"
    elif validator in {"oneOf", "anyOf", "allOf"}:
        expected = validator
        reason = "value does not match the required schema alternative"

    message = f"path={_error_path(error.absolute_path)} expected={expected}; reason={reason}"
    return message[:_MAX_ARGUMENT_ERROR_LENGTH]


def _move_flat_experiment_decisions(arguments, schema):
    """Move only declared decision fields for the structured recipe batch."""
    root_properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    batch_schema = root_properties.get("experiments", {}) if isinstance(root_properties, dict) else {}
    experiment_schema = batch_schema.get("items", {}) if isinstance(batch_schema, dict) else {}
    experiment_properties = experiment_schema.get("properties", {}) if isinstance(experiment_schema, dict) else {}
    decision_schema = experiment_properties.get("decision") if isinstance(experiment_properties, dict) else None
    decision_fields = decision_schema.get("properties", {}) if isinstance(decision_schema, dict) else {}
    known_fields = set(decision_fields) if isinstance(decision_fields, dict) else set()
    experiments = arguments.get("experiments") if isinstance(arguments, dict) else None
    if not known_fields or not isinstance(experiments, list):
        return

    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            continue
        flat_fields = known_fields.intersection(experiment)
        if not flat_fields:
            continue
        decision_present = "decision" in experiment
        nested = experiment.get("decision")
        if decision_present and not isinstance(nested, dict):
            continue
        nested = nested if decision_present else {}
        for field in sorted(flat_fields):
            flat_value = experiment[field]
            if field in nested and nested[field] != flat_value:
                raise _ArgumentNormalizationError(
                    f"$.experiments[{index}].decision.{field}",
                    f"conflicting duplicate field {field!r}",
                )
            nested[field] = flat_value
            del experiment[field]
        experiment["decision"] = nested


def _normalize_decision_string_lists(value, schema, depth=0):
    """Wrap non-empty strings only for the three declared decision string lists."""
    if not isinstance(schema, dict) or depth >= _MAX_CONTAINER_DECODE_DEPTH:
        return value

    properties = schema.get("properties", {})
    if isinstance(value, dict) and isinstance(properties, dict):
        decision = value.get("decision")
        decision_schema = properties.get("decision")
        decision_properties = decision_schema.get("properties", {}) if isinstance(decision_schema, dict) else {}
        if isinstance(decision, dict) and isinstance(decision_properties, dict):
            for field in ("observations", "alternatives", "risks"):
                field_schema = decision_properties.get(field)
                item_schema = field_schema.get("items", {}) if isinstance(field_schema, dict) else {}
                if (_schema_types(field_schema) == {"array"} and _schema_types(item_schema) == {"string"}
                        and isinstance(decision.get(field), str) and decision[field]):
                    decision[field] = [decision[field]]

        additional = schema.get("additionalProperties")
        for key, item in list(value.items()):
            child_schema = properties.get(key)
            if child_schema is None and isinstance(additional, dict):
                child_schema = additional
            if isinstance(child_schema, dict):
                value[key] = _normalize_decision_string_lists(item, child_schema, depth + 1)
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                value[index] = _normalize_decision_string_lists(item, item_schema, depth + 1)
    return value


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.hidden = set()

    def register(self, name, description, parameters, handler, *, advertised=True):
        if name in self.tools:
            raise ValueError(f"Duplicate tool: {name}")
        Draft202012Validator.check_schema(parameters)
        self.tools[name] = (description, parameters, handler)
        if not advertised:
            self.hidden.add(name)

    def schemas(self):
        return [{"type": "function", "function": {"name": name, "description": item[0], "parameters": item[1]}} for name, item in self.tools.items() if name not in self.hidden]

    def execute(self, name, arguments, *, allow_hidden=False):
        if name in self.hidden and not allow_hidden:
            return {"ok": False, "error": {"code": "UNAVAILABLE_TOOL", "message": "Use an advertised tool for this experiment mode"}}
        if name not in self.tools:
            return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}}
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else copy.deepcopy(arguments)
        except (ValueError, TypeError, RecursionError):
            return {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": "path=$ expected=JSON arguments; reason=malformed JSON"}}
        schema = self.tools[name][1]
        parsed = _decode_schema_containers(parsed, schema)
        if name == "run_model_experiments":
            try:
                _move_flat_experiment_decisions(parsed, schema)
            except _ArgumentNormalizationError as exc:
                return {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": str(exc)[:_MAX_ARGUMENT_ERROR_LENGTH]}}
        parsed = _normalize_decision_string_lists(parsed, schema)
        errors = list(Draft202012Validator(schema).iter_errors(parsed))
        if errors:
            errors.sort(key=lambda error: (_error_path(error.absolute_path), str(error.validator)))
            return {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": _compact_validation_error(errors[0])}}
        try:
            value = self.tools[name][2](**parsed)
            return {"ok": True, "value": value}
        except Exception as exc:
            return {"ok": False, "error": {"code": "TOOL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:3000]}}


def parse_response_calls(message):
    calls = message.get("tool_calls")
    if calls:
        normalized = []
        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            normalized.append({"id": call.get("id") or f"call_{uuid4().hex}", "type": "function", "function": {"name": function.get("name", ""), "arguments": arguments}})
        return normalized
    content = message.get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    if "{" not in content or "}" not in content:
        raise ValueError("Return a tool call or a JSON object containing tool and arguments")
    try:
        action = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except (ValueError, TypeError) as exc:
        raise ValueError("Response does not contain valid JSON tool arguments") from exc
    if not isinstance(action, dict):
        raise ValueError("A tool action must be an object")
    name = action.get("tool", action.get("action", action.get("name")))
    if name is None and "experiments" in action:
        proposals = action["experiments"]
        typed = isinstance(proposals, list) and proposals and all(isinstance(p, dict) and "base_trial_id" in p for p in proposals)
        name = "run_model_experiments" if typed else "run_experiments"
    if name is None and "code" in action:
        name = "run_experiment"
    if name is None and "reason" in action:
        name = "finish_research"
    arguments = action.get("arguments", {key: value for key, value in action.items() if key not in {"tool", "action", "name"}})
    if not isinstance(name, str):
        raise ValueError("A tool name is required")
    return [{"id": f"call_{uuid4().hex}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]


def validate_candidate_code(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 64000:
        raise ValueError("Code must contain between 1 and 64000 characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python: {exc.msg} on line {exc.lineno}") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "build_model" for node in tree.body):
        raise ValueError("Define build_model(context) returning an unfitted estimator or Pipeline")
    allowed = {"sklearn", "catboost", "lightgbm", "xgboost", "numpy", "pandas", "scipy", "math", "functools", "collections", "typing", "itertools", "operator", "statistics", "warnings", "re", "json"}
    forbidden_calls = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals", "getattr", "setattr", "delattr", "vars", "exit", "quit"}
    forbidden_attrs = {"read_csv", "read_table", "read_fwf", "read_parquet", "read_pickle", "read_json", "read_excel", "read_html", "read_xml", "read_feather", "read_hdf", "read_orc", "read_sas", "read_spss", "read_stata", "read_sql", "read_sql_query", "read_sql_table", "to_csv", "to_pickle", "load", "loadtxt", "genfromtxt", "fromregex", "fromfile", "tofile", "memmap", "loadmat", "dump", "save", "savez", "system", "popen", "connect", "urlopen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [item.name.split(".")[0] for item in node.names]
            if any(item not in allowed for item in imports):
                raise ValueError("Only modelling-library imports are allowed in candidate code")
        if isinstance(node, ast.ImportFrom) and (node.level or not node.module or node.module.split(".")[0] not in allowed):
            raise ValueError("Only absolute modelling-library imports are allowed")
        if isinstance(node, ast.ImportFrom) and any(item.name in forbidden_attrs for item in node.names):
            raise ValueError("Candidate code cannot import file or network operations")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            raise ValueError(f"Candidate code cannot call {node.func.id}")
        if isinstance(node, ast.Attribute) and ((node.attr.startswith("__") and node.attr != "__init__") or node.attr in forbidden_attrs):
            raise ValueError(f"Candidate code cannot access {node.attr}")
        if isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__", "__file__", "__cached__"}:
            raise ValueError(f"Candidate code cannot access {node.id}")
