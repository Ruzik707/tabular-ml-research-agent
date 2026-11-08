"""Research orchestration; the host owns scoring and model selection."""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

from .contracts import ExperimentResult


def select_best(results, maximize):
    valid = [r for r in results if r.status == "ok" and r.metric is not None and math.isfinite(r.metric)]
    if valid and all(r.trial_id.startswith("trial_") and r.trial_id[6:].isdigit() for r in valid):
        valid.sort(key=lambda r: int(r.trial_id[6:]))
    return (max if maximize else min)(valid, key=lambda r: r.metric) if valid else None


@contextmanager
def _run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process already owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def request_token_bound(messages, tools):
    """Conservative byte-based admission without assuming a model tokenizer.

    Byte-fallback tokenizers cannot need more pieces than the UTF-8 bytes.
    Reserve additional framing space for the provider's chat template.
    """
    content = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
    return len(content.encode("utf-8")) + 1024 + 64 * len(messages)


def recipe_feedback(result):
    if not isinstance(result, dict) or not isinstance(result.get('value'), dict) or not isinstance(result['value'].get('trials'), list):
        return result
    value = result['value']
    rows = [{k:row[k] for k in ('trial_id','status','metrics','error','seconds','selected') if k in row}
            for row in value['trials'] if isinstance(row, dict)]
    return {'ok':result.get('ok'), 'value':{'trials':rows, **{k:value[k] for k in ('selected_trial','best_metric') if k in value}}}


def compact_recipe_history(step, schemas):
    import copy
    from .tools import _decode_schema_containers
    compact = copy.deepcopy(step)
    for message in compact:
        if message.get('role') != 'assistant':
            continue
        message['content'] = (message.get('content') or '')[:600]
        for call in message.get('tool_calls', []):
            function = call.get('function', {})
            if function.get('name') != 'run_model_experiments':
                continue
            try:
                arguments = json.loads(function['arguments']) if isinstance(function.get('arguments'), str) else function['arguments']
                arguments = _decode_schema_containers(arguments, schemas.get(function['name'], {}))
                experiments = arguments.get('experiments')
                if not isinstance(experiments, list):
                    return []  # error feedback carries this malformed call; avoid teaching its bad wire format
                compact_proposals = []
                for row in experiments:
                    if not isinstance(row, dict):
                        return []
                    item = {k:row[k] for k in ('base_trial_id','parameters','preprocessing','fit_options') if k in row}
                    item['hypothesis'] = str(row.get('hypothesis',''))[:300]
                    compact_proposals.append(item)
                function['arguments'] = json.dumps({'experiments':compact_proposals}, ensure_ascii=False)
            except (ValueError, TypeError, KeyError):
                return []
    return compact


class ResearchRun:
    def __init__(self, config):
        initialized_at = config.process_started_at if config.process_started_at is not None else time.monotonic()
        from .data import discover_task
        from .audit import AuditJournal
        from .runtime import Budget

        self.config = config
        self.task = discover_task(config.task_path, seed=config.seed, validation_fraction=config.validation_fraction,
                                  overrides={k: v for k, v in {"target": config.target, "task_type": config.task_type, "metric": config.metric}.items() if v is not None})
        self.directory = config.run_dir or config.task_path.parent / ".automl" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        self.output = config.task_path.parent / "submission.csv"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.state = _read_json(self.state_path, {})
        if self.state and not config.resume:
            raise ValueError("Run directory already contains state; use --resume or a new --run-dir")
        if config.resume and not self.state:
            raise ValueError("No saved state in --run-dir")
        signature = {"data": self.task.data_fingerprint, "seed": config.seed, "validation_fraction": config.validation_fraction,
                     "target": self.task.target, "metric": self.task.metric, "task_type": self.task.task_type}
        self.settings_version = self.state.get("settings_version", 1) if self.state else 2
        if self.settings_version == 2:
            signature["validation_settings"] = {"folds":config.cv_folds,"strategy":config.validation_strategy,"time_column":config.time_column}
        elif config.cv_folds != 1:
            raise ValueError("Legacy run uses holdout; resume with --cv-folds 1 or start a new run")
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if self.state and self.state.get("settings_fingerprint") != self.signature:
            raise ValueError("Input data or split settings changed (fingerprint mismatch); start a new run")
        self.journal = AuditJournal(self.directory / "events.jsonl", secrets=[config.api_key] if config.api_key else [])
        self.results = []
        self.active_trials = {}
        # The event log is authoritative if the process died before state.json was updated.
        completed = {r["trial_id"]: r for r in self.state.get("trials", [])}
        events = [{**event, **event.get("payload", {})} for event in self.journal.read_events()]
        for event in events:
            if event.get("event") == "trial_completed":
                result = event.get("result", {})
                if "trial_id" in result:
                    completed[result["trial_id"]] = result
        self.results = [ExperimentResult.from_dict(self.journal.redact(r)) for r in completed.values()]
        spent = float(self.state.get("elapsed_seconds", 0))
        if self.state.get("active"):
            spent += max(0.0, time.time() - self.state.get("checkpoint_epoch", time.time()))
        spent += time.monotonic() - initialized_at
        self.budget = Budget(config.time_budget, reserve_seconds=config.reserve_seconds, spent_seconds=spent)
        self.tokens_used = int(self.state.get("total_tokens", 0))
        self.steps = int(self.state.get("llm_calls", 0))
        self.feedback = ""
        self.stop_reason = "offline" if config.offline else "time_budget"
        self.finish_requested = False
        self.recent_steps = []
        self.split = None
        self.profile = None
        self.dispatched = {e.get("call_id") for e in events if e.get("event") == "tool_call"}
        self.tool_results = {e.get("call_id"): e.get("result") for e in events if e.get("event") == "tool_result"}
        self._recover_calls(events)
        self._recover_model_state(events)
        self._recover_assessments()
        self._update_reserve(record=False)

    def _recover_model_state(self, events):
        from .contracts import ModelResponse

        requests = {e["request_id"]: e for e in events if e.get("event") == "llm_request"}
        responses = {e["request_id"]: e for e in events if e.get("event") == "llm_response"}
        errors = {e["request_id"]: e for e in events if e.get("event") == "llm_error"}
        consumed = {e.get("request_id") for e in events if e.get("event") in {"llm_response_consumed", "llm_protocol_error"}}
        tokens = 0
        for request_id, request in requests.items():
            bound = request.get("input_token_bound") or request_token_bound(request.get("messages", []), request.get("tools", []))
            if request_id in responses:
                payload = responses[request_id]["response"]
                tokens += responses[request_id].get("charged_tokens") or payload.get("total_tokens") or payload.get("prompt_tokens", 0) + payload.get("completion_tokens", 0) or bound
            elif request_id in errors:
                tokens += errors[request_id].get("charged_tokens", bound)
            else:
                charged = bound + request.get("max_tokens", 0)
                tokens += charged
                self.journal.append("llm_error", request_id=request_id, code="interrupted", error="Outcome unknown after process interruption; request not replayed", charged_tokens=charged)
        self.tokens_used = max(self.tokens_used, tokens)
        self.steps = max(self.steps, len(requests))
        self.pending_responses = [(request_id, ModelResponse(**event["response"])) for request_id, event in responses.items() if request_id not in consumed]
        for event in events:
            if event.get("event") == "tool_result" and event.get("name") == "finish_research" and event.get("result", {}).get("value", {}).get("finished"):
                self.finish_requested = True
                self.stop_reason = "model_finished"

    def _recover_calls(self, events):
        settled = {e.get("call_id") for e in events if e.get("event") == "tool_result"}
        for event in events:
            if event.get("event") == "tool_call" and event.get("call_id") not in settled:
                result = {"ok": False, "error": {"code": "INTERRUPTED", "message": "Previous process ended before a durable result; not replayed"}}
                self.journal.append("tool_result", call_id=event.get("call_id"), name=event.get("name"), result=result)
                self.tool_results[event.get("call_id")] = result
        finished = {r.trial_id for r in self.results}
        for event in events:
            if event.get("event") == "trial_started" and event.get("trial_id") not in finished:
                trial_id = event["trial_id"]
                result = ExperimentResult(trial_id, "interrupted", str(self.directory / trial_id), event.get("code_hash", ""), event.get("hypothesis", ""), error="Outcome unknown after interrupted process; not replayed", is_baseline=event.get("is_baseline", False), decision=event.get("decision", {}), origin=event.get("origin"), recipe=event.get("recipe"), recipe_hash=event.get("recipe_hash"), parent_trial_id=event.get("parent_trial_id"))
                self.results.append(result)
                self.journal.append("trial_completed", result=result.to_dict())
                finished.add(trial_id)

    def _recover_assessments(self):
        """Rebuild missing host explanations from completed, measured trials.

        A crash can land between the durable result and its audit assessment.
        Replaying this pure selection rule never reruns training or an LLM.
        """
        from .strategy import assess_experiment

        events = self.journal.read_events()
        assessed = {e["payload"].get("trial_id") for e in events if e["event"] == "experiment_assessed"}
        ledger = {}
        for event in events:
            if event["event"] != "trial_completed":
                continue
            result = ExperimentResult.from_dict(event["payload"]["result"])
            previous = select_best([r for key, r in ledger.items() if key != result.trial_id], self.task.maximize)
            if result.trial_id not in assessed:
                assessment = assess_experiment(result, previous, self.task.maximize, self.task.metric)
                self.journal.append("experiment_assessed", **assessment, recovered=True, completion_seq=event["seq"])
                assessed.add(result.trial_id)
            ledger[result.trial_id] = result

    def checkpoint(self, active=True):
        from .journal import atomic_write_json

        best = select_best(self.results, self.task.maximize)
        state = {"version": 1, "settings_version": self.settings_version, "data_fingerprint": self.task.data_fingerprint, "settings_fingerprint": self.signature,
                 "elapsed_seconds": self.budget.elapsed(), "checkpoint_epoch": time.time(), "active": active,
                 "total_tokens": self.tokens_used, "llm_calls": self.steps, "trials": [r.to_dict() for r in self.results],
                 "selected_trial": best.trial_id if best else None, "stop_reason": self.stop_reason,
                 "active_trials": list(self.active_trials), "reserve_plan": self.reserve_plan,
                 "validation_settings": {"folds":self.config.cv_folds,"strategy":self.config.validation_strategy,"time_column":self.config.time_column}}
        state = self.journal.redact(state)
        atomic_write_json(self.state_path, state)
        atomic_write_json(self.directory / "experiments.json", {"trials": state["trials"], "selected_trial": state["selected_trial"]})

    def _emergency_submission(self):
        import numpy as np
        import pandas as pd
        from .evaluation import publish_submission
        from .journal import atomic_write_json

        y = pd.read_csv(self.task.train_path, usecols=[self.task.target])[self.task.target]
        constant = y.mode().iloc[0] if self.task.task_type == "classification" else float(y.mean())
        rows = len(pd.read_csv(self.task.test_path, usecols=[self.task.feature_columns[0]]))
        path = self.directory / "fallback_predictions.csv"
        pd.DataFrame({self.task.target: [constant] * rows}).to_csv(path, index=False)
        if getattr(self.task, 'output_kind', 'labels') in {'binary_probability', 'class_probabilities'}:
            classes = list(self.task.classes)
            counts = y.value_counts(normalize=True)
            prior = np.asarray([float(counts.get(label, 0.)) for label in classes])
            np.save(self.directory/'test_probabilities.npy', np.tile(prior, (rows, 1)), allow_pickle=False)
            atomic_write_json(self.directory/'probability_classes.json', {'classes':classes})
        if not self.config.resume or not self.output.exists():
            publish_submission(self.task, path, self.output)
            self.journal.append("fallback_written", submission_path=str(self.output), strategy="mode" if self.task.task_type == "classification" else "mean")

    def _restore_best_submission(self):
        from .evaluation import publish_submission

        while (best := select_best(self.results, self.task.maximize)) is not None:
            try:
                source = Path(best.artifact_dir) / "code.py"
                if hashlib.sha256(source.read_bytes()).hexdigest() != best.code_hash:
                    raise ValueError("Candidate source checksum changed")
                publish_submission(self.task, Path(best.prediction_path), self.output)
                self.journal.append("candidate_restored", trial_id=best.trial_id, metric=best.metric, submission_path=str(self.output))
                return
            except (OSError, ValueError, TypeError) as exc:
                best.status = "invalid_artifact"
                best.error = str(exc)
                self.journal.append("trial_completed", result=best.to_dict())

    def _dispatch_response(self, request_id, response, registry, *, allow_legacy=False):
        from .tools import parse_response_calls

        try:
            calls = parse_response_calls(response.message)
        except ValueError as exc:
            self.feedback = str(exc)
            self.journal.append("llm_protocol_error", request_id=request_id, error=str(exc))
            self.journal.append("llm_response_consumed", request_id=request_id)
            self.checkpoint()
            return
        for index, call in enumerate(calls):
            original = call["id"]
            call["id"] = original if original in self.dispatched else f"{request_id}_tool_{index}"
        step_messages = [{"role": "assistant", "content": response.message.get("content") or "", "tool_calls": calls}]
        history_safe = True
        for call in calls:
            function = call["function"]
            try:
                logged_arguments = json.loads(function["arguments"])
                if not isinstance(logged_arguments, dict):
                    history_safe = False
            except ValueError:
                logged_arguments = function["arguments"]
                history_safe = False
            if call["id"] in self.dispatched:
                result = self.tool_results.get(call["id"], {"ok": False, "error": {"code": "INTERRUPTED", "message": "Not replayed"}})
            else:
                self.journal.append("tool_call", call_id=call["id"], name=function["name"], arguments=logged_arguments)
                self.dispatched.add(call["id"])
                self.checkpoint()
                if self.finish_requested or self.tokens_used >= self.config.max_tokens:
                    code = "TOKEN_BUDGET" if self.tokens_used >= self.config.max_tokens else "CANCELLED"
                    result = {"ok": False, "error": {"code": code, "message": "Research has stopped; tool was not executed"}}
                else:
                    result = registry.execute(function["name"], function["arguments"], allow_hidden=allow_legacy)
                self.journal.append("tool_result", call_id=call["id"], name=function["name"], result=result)
                self.tool_results[call["id"]] = result
            from .prompts import prompt_tool_result
            value = recipe_feedback(result) if self.config.experiment_mode == 'recipe' else prompt_tool_result(result)
            self.feedback = json.dumps(value, ensure_ascii=False, default=str)
            step_messages.append({"role": "tool", "tool_call_id": call["id"], "name": function["name"], "content": self.feedback})
            self.checkpoint()
        if history_safe:
            if self.config.experiment_mode == 'recipe':
                step_messages = compact_recipe_history(step_messages, {name:item[1] for name,item in registry.tools.items()})
            if step_messages:
                self.recent_steps.append(step_messages)
        self.journal.append("llm_response_consumed", request_id=request_id)
        self.checkpoint()

    def _next_trial_id(self):
        indices = [int(p.name.split("_")[1]) for p in self.directory.glob("trial_*") if p.is_dir() and p.name.split("_")[1].isdigit()]
        return f"trial_{max(indices, default=0) + 1:04d}"

    def _update_reserve(self, record=True):
        from .budgeting import plan_finalization
        best = select_best(self.results, self.task.maximize)
        plan = plan_finalization(total_seconds=self.config.time_budget, validation_fraction=self.task.validation_fraction,
                                 best=best, train_rows=(self.profile or {}).get('train_shape', [None])[0])
        if getattr(self, 'split', None) and self.split.get('mode') == 'evaluate_cv':
            valid = [r for r in self.results if r.status == 'ok' and r.metric is not None]
            valid.sort(key=lambda r:-r.metric if self.task.maximize else r.metric)
            from .ensemble import ensemble_shortlist
            costs = []
            for result in ensemble_shortlist(valid):
                estimate = result.model_details.get('training', {}).get('estimated_refit_seconds')
                if isinstance(estimate, (int,float)) and estimate > 0:
                    costs.append(estimate)
            if costs:
                projected = sum(costs)*1.5 + plan['startup_guard_seconds'] + plan['overhead_seconds']
                plan['potential_ensemble_fit_seconds'] = sum(costs)*1.5
                plan['reserve_seconds'] = min(self.config.time_budget, max(plan['reserve_seconds'], projected))
        previous = getattr(self, 'reserve_plan', None)
        self.reserve_plan = plan
        self.budget.reserve_seconds = plan['reserve_seconds']
        if record and previous != plan:
            self.journal.append('reserve_updated', previous_reserve=previous.get('reserve_seconds') if previous else None,
                                plan=plan, remaining_seconds=round(self.budget.remaining(), 3))

    def _prepare_trial(self, hypothesis, code, is_baseline=False, decision=None, origin='llm', recipe=None, recipe_hash=None, parent_trial_id=None):
        from .journal import atomic_write_json
        from .tools import validate_candidate_code
        if len(self.results) + len(self.active_trials) >= self.config.trial_limit:
            raise ValueError('Experiment budget exhausted')
        remaining = self.budget.research_remaining()
        if remaining <= 2:
            raise ValueError('Research deadline reached; finalization time is reserved')
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if any(r.code_hash == code_hash and (recipe_hash is None or r.recipe_hash == recipe_hash)
               for r in [*self.results, *self.active_trials.values()]):
            raise ValueError('Duplicate code: propose a different hypothesis or fix the previous error')
        trial_id = self._next_trial_id()
        directory = self.directory / trial_id
        directory.mkdir()
        (directory / 'code.py').write_text(code, encoding='utf-8')
        result = ExperimentResult(trial_id, 'error', str(directory), code_hash, hypothesis,
                                  is_baseline=is_baseline, decision=decision or {}, origin=origin, recipe=recipe, recipe_hash=recipe_hash, parent_trial_id=parent_trial_id)
        recorded = self.journal.append('decision_recorded', trial_id=trial_id, source='harness' if is_baseline else origin,
                                       hypothesis=hypothesis, decision=result.decision, explanation_provided=bool(result.decision))
        result.decision = recorded['payload']['decision']
        result.hypothesis = recorded['payload']['hypothesis']
        atomic_write_json(directory / 'decision.json', recorded['payload'])
        if recipe is not None:
            atomic_write_json(directory / 'recipe.json', self.journal.redact({'recipe': recipe, 'recipe_hash': recipe_hash, 'parent_trial_id': parent_trial_id}))
        self.active_trials[trial_id] = result
        self.journal.append('trial_started', trial_id=trial_id, code_hash=code_hash, hypothesis=result.hypothesis,
                            is_baseline=is_baseline, origin=origin, decision=result.decision, recipe=result.recipe, recipe_hash=result.recipe_hash, parent_trial_id=result.parent_trial_id)
        self.checkpoint()
        try:
            validate_candidate_code(code)
        except ValueError as exc:
            result.status, result.error = 'invalid_code', str(exc)[-4000:]
            return result, None
        manifest = {'mode': 'evaluate', 'context': self.task.model_context(self.config.n_jobs),
                    'inputs': {key: self.split[key] for key in ('train_features', 'train_target', 'validation_features', 'test_features')},
                    'code_path': str(directory / 'code.py'), 'artifact_dir': str(directory), 'save_model': True,
                    'fit_options': (recipe or {}).get('fit_options', {})}
        if self.split.get('mode') == 'evaluate_cv':
            manifest.update(mode='evaluate_cv', cv_plan=self.split['cv_plan'], preflight=recipe is None)
            manifest['inputs'].pop('validation_features')
        atomic_write_json(directory / 'manifest.json', manifest)
        from .budgeting import trial_time_cap
        timeout = trial_time_cap(total_seconds=self.config.time_budget, research_seconds=remaining,
                                 requested_cap=self.config.trial_timeout)
        return result, {'argv': [sys.executable, '-m', 'automl_agent.worker', '--manifest', str(directory / 'manifest.json')],
                        'cwd': directory, 'timeout': timeout, 'n_jobs': self.config.n_jobs,
                        'memory_limit_mb': self.config.memory_limit_mb, 'env': {'PYTHONPATH': str(Path(__file__).resolve().parent.parent)}}

    def _commit_trial(self, result, process=None):
        from .evaluation import publish_submission, score_trial
        from .journal import atomic_write_json
        from .strategy import assess_experiment
        previous_best = select_best(self.results, self.task.maximize)
        directory = Path(result.artifact_dir)
        if process is not None:
            atomic_write_json(directory / 'process.json', self.journal.redact(process.to_dict()))
            result.seconds, result.status = process.seconds, process.status
            if process.status == 'ok':
                try:
                    result.model_details = _read_json(directory / 'worker_result.json', {}).get('estimator_details', {})
                    result.metrics = score_trial(self.task, self.split, directory)
                    result.metric = float(result.metrics[self.task.metric])
                    from .diagnostics import validation_diagnostics
                    try:
                        result.diagnostics = validation_diagnostics(self.task, self.split, directory)
                        from .validation import oof_metric_summary
                        result.diagnostics['validation'] = oof_metric_summary(self.task, self.split, directory)
                        result.diagnostics['feature_importance'] = result.model_details.get('feature_importance', {'status':'unavailable'})
                    except Exception as exc:
                        result.diagnostics = {'unavailable': type(exc).__name__}
                    if not math.isfinite(result.metric):
                        raise ValueError('Evaluator returned a non-finite metric')
                    result.prediction_path = str(directory / 'test_predictions.csv')
                    publish_submission(self.task, Path(result.prediction_path), directory / 'submission.csv')
                    if (directory / 'model.joblib').exists():
                        result.model_path = str(directory / 'model.joblib')
                except Exception as exc:
                    result.status = 'invalid_output'
                    result.error = f'{type(exc).__name__}: {exc}'[-4000:]
            else:
                result.error = (process.stderr or process.stdout or process.status)[-4000:]
        result = ExperimentResult.from_dict(self.journal.redact(result.to_dict()))
        self.active_trials.pop(result.trial_id, None)
        self.results.append(result)
        self.results.sort(key=lambda r: r.trial_id)
        completion = self.journal.append('trial_completed', result=result.to_dict())
        self.journal.append('experiment_assessed', **assess_experiment(result, previous_best, self.task.maximize, self.task.metric), completion_seq=completion['seq'])
        best = select_best(self.results, self.task.maximize)
        if best is result:
            publish_submission(self.task, Path(result.prediction_path), self.output)
            self.journal.append('candidate_selected', trial_id=result.trial_id, metric=result.metric, submission_path=str(self.output))
        self._update_reserve()
        self.checkpoint()
        return result

    def experiment_batch(self, experiments, is_baseline=False, origin='llm'):
        from .scheduler import run_parallel_processes
        outcomes = [None] * len(experiments)
        prepared = []
        for index, proposal in enumerate(experiments):
            try:
                result, job = self._prepare_trial(proposal['hypothesis'], proposal['code'], is_baseline,
                                                  proposal.get('decision'), origin, proposal.get('recipe'), proposal.get('recipe_hash'), proposal.get('parent_trial_id'))
                if job is None:
                    outcomes[index] = self._commit_trial(result)
                else:
                    prepared.append((index, result, job))
            except ValueError as exc:
                outcomes[index] = {'error': str(exc)}
        if not prepared:
            return outcomes
        ids = [result.trial_id for _, result, _ in prepared]
        self.journal.append('batch_started', trial_ids=ids, origin=origin, parallel_trials=self.config.parallel_trials,
                            threads_per_trial=self.config.n_jobs, memory_limit_mb=self.config.memory_limit_mb)
        try:
            with closing(run_parallel_processes([job for _, _, job in prepared], max_workers=self.config.parallel_trials,
                                                 research_remaining=self.budget.research_remaining)) as completed:
                for index, process in completed:
                    output_index, result, _ = prepared[index]
                    outcomes[output_index] = self._commit_trial(result, process)
        except KeyboardInterrupt:
            self.journal.append('batch_interrupted', trial_ids=list(self.active_trials))
            self.checkpoint()
            raise
        self.journal.append('batch_completed', trial_ids=ids, remaining_seconds=round(self.budget.research_remaining(), 3))
        return outcomes

    def experiment(self, hypothesis, code, is_baseline=False, decision=None):
        result = self.experiment_batch([{'hypothesis': hypothesis, 'code': code, 'decision': decision}],
                                       is_baseline=is_baseline, origin='baseline' if is_baseline else 'llm')[0]
        if isinstance(result, dict):
            raise ValueError(result['error'])
        return result

    def _effective_recipe(self, result):
        """Resolve native defaults from the measured base before applying edits."""
        from .recipes import normalize_recipe
        recipe = normalize_recipe(result.recipe, self.task.task_type)
        if recipe['family'] == 'CatBoost':
            actual = result.model_details.get('parameters', {})
            for key in ('learning_rate', 'depth', 'l2_leaf_reg', 'border_count', 'auto_class_weights'):
                if key in actual and (key != 'auto_class_weights' or self.task.task_type == 'classification'):
                    recipe['parameters'][key] = actual[key]
        return normalize_recipe(recipe, self.task.task_type)

    def _llm_model_experiments(self, experiments, _source='llm'):
        from .recipes import normalize_recipe, recipe_code, recipe_fingerprint
        if not 1 <= len(experiments) <= 2:
            raise ValueError('Batch must contain 1 or 2 independent candidates; workers execute the queue within resource limits')
        known = set()
        for prior in [*self.results, *self.active_trials.values()]:
            if prior.recipe:
                known.add(recipe_fingerprint(prior.recipe, self.task.task_type))
                known.add(recipe_fingerprint(self._effective_recipe(prior), self.task.task_type))
        proposals, positions = [], []
        outcomes = [None] * len(experiments)
        for position, experiment in enumerate(experiments):
            try:
                base = next((r for r in self.results if r.trial_id == experiment.get('base_trial_id')), None)
                if base is None or base.status != 'ok' or not base.recipe:
                    raise ValueError('base_trial_id must name a successful measured recipe trial from this run')
                recipe = self._effective_recipe(base)
                parameters, preprocessing = experiment.get('parameters', {}), experiment.get('preprocessing', {})
                if not isinstance(parameters, dict) or not isinstance(preprocessing, dict):
                    raise ValueError('parameters and preprocessing must be JSON objects')
                recipe['parameters'].update(parameters)
                recipe['preprocessing'].update(preprocessing)
                if 'fit_options' in experiment:
                    if not isinstance(experiment['fit_options'], dict):
                        raise ValueError('fit_options must be a JSON object')
                    recipe.setdefault('fit_options', {}).update(experiment['fit_options'])
                recipe = normalize_recipe(recipe, self.task.task_type)
                if recipe['preprocessing'].get('add_missing_indicators') and not any(c not in self.task.categorical_columns for c in self.task.feature_columns):
                    raise ValueError('Missing indicators require numeric columns; this task has only categorical features')
                fingerprint = recipe_fingerprint(recipe, self.task.task_type)
                if fingerprint in known:
                    raise ValueError('Duplicate recipe: this configuration is already measured, queued or interrupted; propose a real change')
                known.add(fingerprint)
                proposals.append({'hypothesis': experiment['hypothesis'], 'decision': experiment.get('decision'),
                                  'code': recipe_code(recipe, self.task.task_type, self.config.n_jobs),
                                  'recipe': recipe, 'recipe_hash': fingerprint, 'parent_trial_id': base.trial_id})
                positions.append(position)
            except (ValueError, TypeError, KeyError) as exc:
                outcomes[position] = {'status': 'invalid_recipe', 'error': str(exc)}
                self.journal.append('recipe_rejected', hypothesis=experiment.get('hypothesis', ''),
                                    base_trial_id=experiment.get('base_trial_id'), error=str(exc))
        if proposals:
            for position, result in zip(positions, self.experiment_batch(proposals, origin=_source)):
                outcomes[position] = result
        best = select_best(self.results, self.task.maximize)
        return {'trials': [({'trial_id': r.trial_id, 'status': r.status, 'metrics': r.metrics, 'error': r.error,
                            'seconds': r.seconds, 'recipe': r.recipe, 'recipe_hash': r.recipe_hash,
                            'parent_trial_id': r.parent_trial_id, 'model_details': r.model_details, 'diagnostics': r.diagnostics,
                            'selected': best is r} if isinstance(r, ExperimentResult) else r) for r in outcomes],
                'best_metric': best.metric if best else None, 'selected_trial': best.trial_id if best else None}

    def _family_comparison(self, strategy):
        from .families import family_candidates
        from .journal import atomic_write_json
        from .modeling import baseline_code
        from .strategy import baseline_decision
        candidates = family_candidates(self.task, self.profile, n_jobs=self.config.n_jobs, limit=self.config.family_trials)
        from .recipes import default_recipe, recipe_fingerprint
        for proposal in candidates:
            proposal['recipe'] = default_recipe(proposal['family'], self.task.task_type)
            proposal['recipe_hash'] = recipe_fingerprint(proposal['recipe'], self.task.task_type)
        if self.config.cv_folds > 1:
            from .recipes import initial_research_recipes, recipe_code
            candidates = []
            for recipe in initial_research_recipes(self.task.task_type, self.profile['available_packages'])[:self.config.family_trials]:
                family = recipe['family']
                candidates.append({'family':family, 'recipe':recipe, 'recipe_hash':recipe_fingerprint(recipe, self.task.task_type),
                    'code':recipe_code(recipe, self.task.task_type, self.config.n_jobs),
                    'hypothesis':f'Сравнить {family} на общем OOF; остановка бустинга по внутренней validation.',
                    'decision':{'model_family':family, 'rationale':'Стартовое сравнение разных семейств на воспроизводимой CV.',
                                'expected_effect':'Качество неизвестно до проверки.', 'risks':['Адаптивный выбор по OOF может переоценивать качество.']}})
        # Preserve two LLM attempt slots. If a complete requested comparison
        # cannot fit, start from the legacy baseline and say why in the log.
        limited = not self.config.offline and self.config.trial_limit < len(candidates) + 2
        if self.config.family_trials == 1 or limited:
            if self.config.experiment_mode == 'recipe':
                candidates = candidates[:1]
            else:
                candidates = [{'family': baseline_decision(strategy)['model_family'], 'hypothesis': 'Проверить надёжный baseline перед LLM-исследованием',
                               'code': baseline_code(self.task, n_jobs=self.config.n_jobs), 'decision': baseline_decision(strategy)}]
        attempted = {r.decision.get('model_family') for r in self.results if r.is_baseline}
        proposed = [p for p in candidates if p['family'] not in attempted]
        proposed = proposed[:None if self.config.max_trials is None else max(0, self.config.max_trials - len(self.results))]
        self.journal.append('family_comparison_started', families=[p['family'] for p in candidates],
                            planned=len(proposed), limited_by_trial_budget=limited, split_fingerprint=self.split['split_fingerprint'])
        warmup_started = self.budget.elapsed()
        warmup_window = self.budget.research_remaining() * 0.5
        for proposal in proposed:
            if self.budget.research_remaining() <= 2:
                break
            if any(r.status == 'ok' and r.is_baseline for r in self.results) and self.budget.elapsed() - warmup_started >= warmup_window:
                self.journal.append('family_comparison_deferred', family=proposal['family'], reason='preserve_research_window')
                break
            self.experiment_batch([proposal], is_baseline=True, origin='family')
            # Time-first runs try the cheap host feature/weight chain before a
            # costly later family. Explicit trial caps retain their LLM slots.
            if (self.config.max_trials is None and not self.config.offline
                    and self.budget.elapsed() - warmup_started < warmup_window):
                self._host_feature_step()
        trials = [r for r in self.results if r.is_baseline]
        best = select_best(trials, self.task.maximize)
        comparison = {'split_fingerprint': self.split['split_fingerprint'], 'metric': self.task.metric,
                      'maximize': self.task.maximize, 'limited_by_trial_budget': limited,
                      'trials': [{'family': r.decision.get('model_family', 'legacy'), 'trial_id': r.trial_id,
                                  'status': r.status, 'metric': r.metric, 'seconds': r.seconds, 'model_details': r.model_details, 'recipe': r.recipe, 'recipe_hash': r.recipe_hash} for r in trials],
                      'selected_trial': best.trial_id if best else None}
        atomic_write_json(self.directory / 'family_comparison.json', self.journal.redact(comparison))
        self.profile["family_comparison"] = comparison
        self.journal.append('family_comparison_completed', comparison=comparison)

    def _llm_experiments(self, experiments):
        if not 1 <= len(experiments) <= 2:
            raise ValueError('Batch must contain 1 or 2 candidates; concurrency is controlled by the worker queue')
        results = self.experiment_batch(experiments)
        best = select_best(self.results, self.task.maximize)
        return {'trials': [({'trial_id': r.trial_id, 'status': r.status, 'metrics': r.metrics, 'error': r.error,
                            'model_details': r.model_details, 'diagnostics': r.diagnostics, 'selected': best is r} if isinstance(r, ExperimentResult) else r) for r in results],
                'best_metric': best.metric if best else None, 'selected_trial': best.trial_id if best else None}

    def bounded_search(self, limit):
        from .search import next_search_proposal
        for _ in range(limit):
            if len(self.results) >= self.config.trial_limit or self.budget.research_remaining() <= 2:
                break
            step = sum(r.origin == 'search' for r in self.results)
            if self._host_feature_step():
                continue
            proposal = next_search_proposal(self.results, self.task.task_type, self.task.maximize, self.config.seed, step)
            if proposal is None:
                break
            self.journal.append('search_proposal', proposal=proposal, step=step)
            self._llm_model_experiments([proposal], _source='search')

    def _host_feature_step(self):
        from .search import next_feature_or_weight_proposal
        proposal = next_feature_or_weight_proposal(
            self.results, self.task.task_type, self.task.metric, self.task.feature_columns,
            self.task.categorical_columns, self.config.seed, sum(r.origin == 'search' for r in self.results))
        if proposal is None or self.budget.research_remaining() <= 2:
            return False
        self.journal.append('search_proposal', proposal=proposal, source='host_feature_chain')
        before = len(self.results)
        self._llm_model_experiments([proposal], _source='search')
        return len(self.results) > before

    def _complete_during_search(self, provider, messages, tools, *, timeout, max_tokens):
        """Keep host state on this thread while HTTP and bounded fits overlap."""
        from concurrent.futures import Future, TimeoutError as FutureTimeout
        import threading
        from .providers import ProviderError

        future = Future()
        cancellation = provider.new_cancel_token()
        def complete():
            if not future.set_running_or_notify_cancel():
                return
            try:
                with provider.request_scope(cancellation):
                    response = provider.complete(messages, tools, timeout=timeout, max_tokens=max_tokens)
                future.set_result(response)
            except BaseException as exc:
                future.set_exception(exc)
        thread = threading.Thread(target=complete, name='automl-llm-request', daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        exhausted = False
        try:
            while True:
                try:
                    return future.result(timeout=0.03)
                except FutureTimeout:
                    if future.done():
                        raise  # A provider exception is not a pending-Future timeout.
                    if self.budget.research_remaining() <= 2 or time.monotonic() >= deadline:
                        raise ProviderError('LLM research window exhausted', code='cancelled', retryable=False)
                    # Keep two slots for the response when the user set a cap.
                    can_search = (self.config.experiment_mode == 'recipe' and not exhausted
                                  and len(self.results) + 2 < self.config.trial_limit)
                    if can_search:
                        before = len(self.results)
                        self.bounded_search(1)
                        exhausted = len(self.results) == before
                        if not exhausted:
                            self.journal.append('llm_wait_search_completed', trial_ids=[r.trial_id for r in self.results[before:]])
        finally:
            if not future.done():
                cancellation.cancel()
                future.cancel()
                thread.join(timeout=0.2)

    def _llm_experiment(self, hypothesis, code, decision=None):
        result = self.experiment(hypothesis, code, decision=decision)
        if result.status != "ok":
            raise ValueError(f"{result.trial_id} {result.status}: {result.error}")
        best = select_best(self.results, self.task.maximize)
        return {"trial_id": result.trial_id, "metrics": result.metrics, "seconds": result.seconds,
                "selected": best.trial_id == result.trial_id, "best_metric": best.metric, "model_details": result.model_details, "diagnostics": result.diagnostics}

    def _finish(self, reason):
        if not any(r.status == "ok" and not r.is_baseline for r in self.results):
            raise ValueError("Run at least one successful generated experiment before finishing")
        if self.config.max_trials is None and self.config.max_steps is None:
            generated = [r for r in self.results if r.status == 'ok' and not r.is_baseline]
            if self.budget.research_remaining() > max(30., 1.5*max((r.seconds for r in generated[-4:]), default=0)):
                raise ValueError('Time-first research: continue evidence-driven experiments while research time remains, including after non-improving trials. Finish when another measured experiment no longer fits before the finalization reserve.')
        self.finish_requested = True
        self.stop_reason = "model_finished"
        self.journal.append("research_finished", source="llm", reason=reason)
        return {"finished": True, "reason": reason}

    def research(self):
        from .prompts import build_messages
        from .providers import OllamaProvider, OpenAICompatibleProvider, ProviderError
        from .tools import ToolRegistry
        from .strategy import DECISION_SCHEMA

        provider_type = OllamaProvider if self.config.provider == "ollama" else OpenAICompatibleProvider
        provider = provider_type(base_url=self.config.base_url, api_key=self.config.api_key, model=self.config.model, extra_body=self.config.extra_body)
        registry = ToolRegistry()
        mode = self.config.experiment_mode
        if mode == "recipe" and not any(r.status == "ok" and r.recipe for r in self.results):
            mode = "code"
        self.journal.append("research_mode_selected", requested=self.config.experiment_mode, effective=mode)
        registry.register("run_experiment", "Test a concrete ML hypothesis with complete build_model(context) Python source.",
                          {"type": "object", "properties": {"hypothesis": {"type": "string", "minLength": 1, "maxLength": 6000}, "code": {"type": "string", "minLength": 1, "maxLength": 64000}, "decision": DECISION_SCHEMA}, "required": ["hypothesis", "code"], "additionalProperties": False}, self._llm_experiment, advertised=mode == "code")
        proposal_schema = registry.tools["run_experiment"][1]
        registry.register("run_experiments", "Evaluate two independent hypotheses in parallel on the same holdout; return separate outcomes.",
                          {"type": "object", "properties": {"experiments": {"type": "array", "minItems": 1,
                           "maxItems": 2, "items": proposal_schema}},
                           "required": ["experiments"], "additionalProperties": False}, self._llm_experiments, advertised=mode == "code")
        model_proposal = {"type": "object", "properties": {
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 6000},
            "base_trial_id": {"type": "string", "pattern": "^trial_[0-9]+$", "maxLength": 40},
            "parameters": {"type": "object", "maxProperties": 16,
                           "description": "Only changed estimator parameters from model_catalog; omitted parameters inherit measured base defaults."},
            "preprocessing": {"type": "object", "properties": {"add_missing_indicators": {"type": "boolean"},
                "feature_ops": {"type": "array", "maxItems": 16, "items": {"type": "object"}}}, "additionalProperties": False},
            "fit_options": {"type": "object", "description": "Changed host fit options: early_stopping_rounds, inner_validation_fraction, sample_weight."},
            "decision": DECISION_SCHEMA}, "required": ["hypothesis", "base_trial_id"], "additionalProperties": False}
        registry.register("run_model_experiments", "Test 1-2 independent parameter/preprocessing edits to measured model recipes; host preserves validated pipeline and inherited defaults.",
                          {"type": "object", "properties": {"experiments": {"type": "array", "minItems": 1,
                           "maxItems": 2, "items": model_proposal}},
                           "required": ["experiments"], "additionalProperties": False}, self._llm_model_experiments,
                          advertised=mode == "recipe")
        registry.register("finish_research", "Stop research after a successful generated candidate; host refits the best model.",
                          {"type": "object", "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": 4000}}, "required": ["reason"], "additionalProperties": False}, self._finish)
        for request_id, response in self.pending_responses:
            self.journal.append("llm_response_recovered", request_id=request_id)
            self._dispatch_response(request_id, response, registry, allow_legacy=True)
        compact = mode == 'recipe'
        while len(self.results) < self.config.trial_limit and self.steps < self.config.step_limit and not self.finish_requested:
            if self.tokens_used >= self.config.max_tokens:
                self.stop_reason = "token_budget"
                break
            if self.budget.research_remaining() <= 1:
                self.stop_reason = "time_budget"
                break
            if compact and mode == "code":
                registry.hidden.add("run_experiments")  # avoid duplicating the full source-code schema under a tight token budget
            best = select_best(self.results, self.task.maximize)
            try:
                messages = build_messages(self.task, self.profile, self.results, best,
                                          {"n_jobs": self.config.n_jobs, "remaining_seconds": round(self.budget.research_remaining(), 2),
                                           "trial_timeout": self.config.trial_timeout, "remaining_trials": None if self.config.max_trials is None else self.config.max_trials - len(self.results),
                                           "parallel_trials": self.config.parallel_trials, "proposal_batch_size": 2,
                                           "reserve_plan": self.reserve_plan, "experiment_mode": mode},
                                          feedback=self.feedback, context_chars=self.config.context_chars, compact=compact)
            except ValueError as exc:
                # Formatting is host preparation, not a sent or billable call.
                # Keep the measured winner eligible for local search and refit.
                self.stop_reason = 'prompt_context_error'
                self.journal.append('llm_context_error', code=self.stop_reason, error=str(exc))
                break
            # Keep complete recent assistant/tool groups. Older attempts remain
            # available through the structured ledger and the durable journal.
            for step_messages in self.recent_steps[-(1 if compact or mode == 'recipe' else 2):]:
                messages.extend(step_messages)
            estimate = request_token_bound(messages, registry.schemas())
            available_tokens = self.config.max_tokens - self.tokens_used - estimate
            if available_tokens < 128:
                if not compact:
                    compact = True
                    self.journal.append("context_compacted", reason="token_budget_admission")
                    continue
                self.stop_reason = "token_budget"
                break
            output_cap = min(self.config.response_tokens, available_tokens)
            self.steps += 1
            request_id = f"request_{self.steps:04d}"
            self.journal.append("llm_request", request_id=request_id, provider=self.config.provider, model=self.config.model,
                                messages=messages, tools=registry.schemas(), max_tokens=output_cap, input_token_bound=estimate)
            self.checkpoint()
            try:
                response = self._complete_during_search(provider, messages, registry.schemas(),
                    timeout=min(self.config.llm_timeout, max(0.1, self.budget.research_remaining() - 2)), max_tokens=output_cap)
            except ProviderError as exc:
                # Failed requests may still be billable; reserve their estimated input.
                charged = estimate + output_cap
                self.tokens_used += charged
                self.journal.append("llm_error", request_id=request_id, code=exc.code, error=str(exc), charged_tokens=charged)
                if exc.code in {"context_overflow", "CONTEXT_WINDOW_EXCEEDED", "context_length_exceeded"} and not compact:
                    compact = True
                    self.feedback = "Context overflow: the host reduced auxiliary context. Return one compact tool call."
                    self.journal.append("context_compacted", reason="provider_context_overflow")
                    continue
                self.stop_reason = "time_budget" if exc.code == 'cancelled' or self.budget.research_remaining() <= 2 else "provider_error"
                break
            charged = response.total_tokens or response.prompt_tokens + response.completion_tokens or estimate + len(json.dumps(response.message).encode("utf-8"))
            self.tokens_used += charged
            self.journal.append("llm_response", request_id=request_id, response=response.to_dict(), charged_tokens=charged)
            self._dispatch_response(request_id, response, registry)
        if self.steps >= self.config.step_limit and not self.finish_requested:
            self.stop_reason = "max_steps"
        if len(self.results) >= self.config.trial_limit:
            self.stop_reason = "max_trials"
        self.journal.append("research_stopped", reason=self.stop_reason, total_tokens=self.tokens_used, elapsed_seconds=round(self.budget.elapsed(), 3))

    def finalize(self, best):
        from .evaluation import publish_submission
        from .journal import atomic_write_json
        from .runtime import run_process
        from .budgeting import can_start_final_fit

        write_margin = max(1.0, self.reserve_plan.get("overhead_seconds", 0.0))
        if best is None or self.budget.remaining() <= write_margin + 0.1:
            return {"status": "skipped", "reason": "No candidate or no remaining time"}
        plan = getattr(self, 'postprocessing_plan', {})
        if plan.get('accepted'):
            estimate = plan.get('estimated_fit_seconds')
            if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate < 0:
                return {"status": "skipped", "reason": "No measured final-ensemble fit window; saved checkpoint retained"}
            required = estimate + self.reserve_plan.get('startup_guard_seconds', 0.) + write_margin
            total = getattr(self.config, 'time_budget', math.inf)
            self.reserve_plan = {**self.reserve_plan, 'source': 'postprocessing',
                'estimated_fit_seconds': float(estimate),
                'required_seconds': required, 'feasible': required <= total,
                'reserve_seconds': min(total, max(required, self.reserve_plan.get('reserve_seconds', 0.)))}
        if not can_start_final_fit(self.budget.remaining(), self.reserve_plan):
            return {"status": "skipped", "reason": "Insufficient full final-fit window; saved checkpoint retained"}
        directory = self.directory / ("final_" + uuid4().hex[:8])
        directory.mkdir()
        source = Path(best.artifact_dir) / "code.py"
        (directory / "code.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = {"mode": "finalize", "context": self.task.model_context(self.config.n_jobs), "inputs": {
            "train_features": self.split["full_train_features"], "train_target": self.split["full_train_target"], "test_features": self.split["test_features"]},
            "code_path": str(directory / "code.py"), "artifact_dir": str(directory), "save_model": True,
            "fit_options": (best.recipe or {}).get("fit_options", {}),
            "iteration_override": best.model_details.get("training", {}).get("final_iterations")}
        plan = getattr(self, 'postprocessing_plan', {})
        if plan.get('accepted'):
            members, weights = [], []
            for index, (trial_id, weight) in enumerate(zip(plan['member_trial_ids'], plan['weights'])):
                if weight <= 0:
                    continue
                result = next(r for r in self.results if r.trial_id == trial_id)
                member_source = Path(result.artifact_dir)/'code.py'
                if hashlib.sha256(member_source.read_bytes()).hexdigest() != result.code_hash:
                    raise ValueError('Ensemble member source checksum changed')
                member_code = directory/f'code_member_{index}.py'
                member_code.write_text(member_source.read_text(encoding='utf-8'), encoding='utf-8')
                members.append({'trial_id':trial_id, 'code_path':str(member_code),
                     'fit_options':(result.recipe or {}).get('fit_options',{}),
                     'iteration_override':result.model_details.get('training',{}).get('final_iterations')})
                weights.append(weight)
            manifest.update(mode='finalize_ensemble', members=members, weights=weights,
                            class_multipliers=plan['class_multipliers'],
                            binary_threshold=plan.get('binary_threshold'), positive_class=plan.get('positive_class'))
        atomic_write_json(directory / "manifest.json", manifest)
        self.journal.append("final_fit_started", trial_id=best.trial_id, artifact_dir=str(directory), reserve_plan=self.reserve_plan, timeout_seconds=max(0.1, self.budget.remaining() - write_margin))
        self.checkpoint()
        process = run_process([sys.executable, "-m", "automl_agent.worker", "--manifest", str(directory / "manifest.json")],
                              cwd=directory, timeout=max(0.1, self.budget.remaining() - write_margin), n_jobs=self.config.n_jobs,
                              memory_limit_mb=self.config.memory_limit_mb, env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)})
        atomic_write_json(directory / "process.json", self.journal.redact(process.to_dict()))
        outcome = {"status": process.status, "artifact_dir": str(directory), "seconds": process.seconds}
        if process.status == "ok":
            try:
                publish_submission(self.task, directory / "test_predictions.csv", self.output)
                outcome["model_details"] = _read_json(directory / "worker_result.json", {}).get("estimator_details", {})
                outcome["model_path"] = str(directory / "model.joblib") if (directory / "model.joblib").exists() else None
            except Exception as exc:
                outcome.update(status="invalid_output", error=str(exc))
        else:
            outcome["error"] = (process.stderr or process.stdout)[-3000:]
        outcome = self.journal.redact(outcome)
        self.journal.append("final_fit_completed", result=outcome)
        if process.status == "cancelled":
            raise KeyboardInterrupt
        return outcome

    def execute(self):
        from .data import prepare_split, profile_task
        from .evaluation import validate_submission
        from .journal import atomic_write_json
        from .modeling import baseline_code
        from .strategy import baseline_decision, plan_research

        if self.config.resume and not self.config.finalize_only and not self.state.get("active") and (self.directory / "report.json").exists():
            validate_submission(self.task, self.output)
            # Older completed runs may predate snapshot redaction. Preserve
            # no-op resume semantics while cleaning their public reports too.
            for name in ("state.json", "experiments.json", "report.json"):
                path = self.directory / name
                if path.exists():
                    atomic_write_json(path, self.journal.redact(_read_json(path)))
            markdown = self.directory / "report.md"
            if markdown.exists():
                markdown.write_text(self.journal.redact(markdown.read_text(encoding="utf-8")), encoding="utf-8")
            report = _read_json(self.directory / "report.json")
            report.setdefault("decision_log", str(self.directory / "agent.log"))
            report.setdefault("decisions_path", str(self.directory / "decisions.jsonl"))
            report = self.journal.redact(report)
            atomic_write_json(self.directory / "report.json", report)
            return report
        self.journal.append("run_resumed" if self.config.resume else "run_started", config=self.config.public_dict(), data_fingerprint=self.task.data_fingerprint)
        self._emergency_submission()
        atomic_write_json(self.directory / "config.json", self.config.public_dict())
        atomic_write_json(self.directory / "task.json", self.journal.redact(self.task.to_dict()))
        versions = {}
        for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "optuna", "torch", "autogluon.tabular", "flaml", "lightautoml", "httpx", "psutil", "cloudpickle"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unavailable"
        atomic_write_json(self.directory / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": versions})
        self.profile = profile_task(self.task)
        self.profile["available_packages"] = versions
        self._update_reserve()
        strategy = plan_research(self.task, self.profile, self.config)
        self.profile["research_strategy"] = strategy
        atomic_write_json(self.directory / "strategy.json", strategy)
        self.journal.append("data_profiled", summary={**strategy["facts"], "task_type": self.task.task_type, "metric": self.task.metric,
                                                   "target": self.task.target, "test_rows": self.profile["test_shape"][0]})
        self.journal.append("research_strategy", strategy=strategy)
        atomic_write_json(self.directory / "profile.json", self.profile)
        if self.config.cv_folds > 1:
            from .data_diagnostics import dataset_diagnostics
            from .validation import prepare_cv_split
            cached = self.directory / "dataset_diagnostics.json"
            drift = _read_json(cached)
            if drift is None:
                drift = dataset_diagnostics(self.task, n_jobs=self.config.n_jobs)
                atomic_write_json(cached, drift)
            self.profile["dataset_diagnostics"] = drift
            self.journal.append("dataset_diagnosed", diagnostics=drift)
            self.split = prepare_cv_split(self.task, self.directory, folds=self.config.cv_folds,
                        strategy=self.config.validation_strategy, time_column=self.config.time_column, diagnostics=drift)
            self.profile["validation_plan"] = {k:self.split[k] for k in ('strategy','fold_count','coverage','time_column','warnings')}
        else:
            self.split = prepare_split(self.task, self.directory)
        atomic_write_json(self.directory / "profile.json", self.profile)
        atomic_write_json(self.directory / "split_manifest.json", self.split)
        self.journal.append("validation_planned", summary={"seed": self.task.seed, "fraction": self.task.validation_fraction,
                            "manifest_path": str(self.directory / "split_manifest.json"), "split_fingerprint": self.split["split_fingerprint"],
                            "policy": self.split.get("strategy", "legacy_holdout"), "fold_count": self.split.get("fold_count", 1), "coverage": self.split.get("coverage"), "warnings": self.split.get("warnings", [])})
        if self.config.resume:
            self._restore_best_submission()
        self.checkpoint()
        try:
            if self.config.finalize_only:
                self.stop_reason = "finalize_only"
                self.journal.append("research_stopped", reason=self.stop_reason, total_tokens=self.tokens_used, elapsed_seconds=round(self.budget.elapsed(), 3))
            if not self.config.finalize_only and self.budget.research_remaining() > 0:
                self._family_comparison(strategy)
            if not self.config.offline and not self.config.finalize_only:
                self.research()
            no_llm_candidate = not any(r.status == 'ok' and r.origin == 'llm' and not r.is_baseline for r in self.results)
            fallback_search = self.stop_reason in {'provider_error', 'prompt_context_error'} or (not self.config.offline and self.steps > 0 and no_llm_candidate)
            if not self.config.finalize_only and (self.config.search_trials or fallback_search):
                self.bounded_search(self.config.search_trials or 4)
            best = select_best(self.results, self.task.maximize)
            self.postprocessing_plan = {'accepted':False, 'reason':'no_time_or_oof'}
            if self.split.get('mode') == 'evaluate_cv' and self.budget.remaining() > self.reserve_plan.get('overhead_seconds', 0) + 2:
                from .ensemble import choose_postprocessing_bounded
                try:
                    self.postprocessing_plan = choose_postprocessing_bounded(self.task, self.split, self.results, self.directory,
                        remaining_seconds=self.budget.remaining(), overhead_seconds=self.reserve_plan.get('overhead_seconds', 5),
                        refit_seconds=self.reserve_plan.get('estimated_fit_seconds') or 0.,
                        startup_guard_seconds=self.reserve_plan.get('startup_guard_seconds', 0.),memory_limit_mb=self.config.memory_limit_mb)
                    if self.postprocessing_plan.get('accepted'):
                        from .evaluation import publish_submission
                        publish_submission(self.task, Path(self.postprocessing_plan['prediction_path']), self.output)
                except Exception as exc:
                    self.postprocessing_plan = {'accepted':False, 'reason':'postprocessing_error', 'error':str(exc)[-1000:]}
                atomic_write_json(self.directory/'postprocessing.json', self.journal.redact(self.postprocessing_plan))
                self.journal.append('postprocessing_selected', plan=self.postprocessing_plan)
                self.checkpoint()
            final_fit = self.finalize(best)
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            self.journal.append("run_interrupted")
            self.checkpoint()
            raise
        except Exception as exc:
            self.stop_reason = "orchestration_error"
            self.journal.append("run_error", error=f"{type(exc).__name__}: {exc}")
            best = select_best(self.results, self.task.maximize)
            final_fit = {"status": "skipped", "reason": str(exc)}
        validation = validate_submission(self.task, self.output)
        self.journal.append("submission_validated", validation=validation, submission_path=str(self.output))
        llm_trials = sum(r.status == "ok" and not r.is_baseline and r.origin in (None, 'llm') for r in self.results)
        status = "offline" if self.config.offline else ("completed" if llm_trials else "degraded")
        report = {"status": status, "stop_reason": self.stop_reason, "task_type": self.task.task_type, "metric": self.task.metric,
                  "best_metric": best.metric if best else None, "selected_trial": best.trial_id if best else None,
                  "trials": [r.to_dict() for r in self.results], "llm_calls": self.steps, "llm_trials": llm_trials,
                  "total_tokens": self.tokens_used, "elapsed_seconds": round(self.budget.elapsed(), 3),
                  "submission_path": str(self.output), "run_dir": str(self.directory), "final_fit": final_fit, "validation": validation,
                  "decision_log": str(self.directory / "agent.log"), "decisions_path": str(self.directory / "decisions.jsonl"),
                  "reserve_plan": self.reserve_plan, "parallel_trials": self.config.parallel_trials,
                  "postprocessing": getattr(self, "postprocessing_plan", {}), "validation_plan": self.profile.get("validation_plan", {}),
                  "threads_per_trial": self.config.n_jobs, "family_comparison_path": str(self.directory / "family_comparison.json")}
        report = self.journal.redact(report)
        atomic_write_json(self.directory / "report.json", report)
        rows = ["# AutoResearch run", "", f"Status: {status}; stop: {self.stop_reason}", f"Metric: {self.task.metric}; best validation: {report['best_metric']}",
                f"Submission: {self.output}", f"Decision log: {self.directory / 'agent.log'}", f"Structured decisions: {self.directory / 'decisions.jsonl'}", "", "| Trial | Status | Validation metric | Seconds |", "|---|---|---:|---:|"]
        rows += [f"| {r.trial_id} | {r.status} | {r.metric} | {r.seconds:.2f} |" for r in self.results]
        for result in self.results:
            rows += ["", f"## {result.trial_id}: {result.hypothesis}", f"Decision: {Path(result.artifact_dir) / 'decision.json'}",
                     f"Actual estimator: {' → '.join(result.model_details.get('estimator_chain', [])) or 'not recorded'}"]
        rows += ["", "Validation uses the recorded shared split/CV plan. Adaptive model selection can overestimate generalization. The final refit uses all training rows.",
                 "Generated code runs in a supervised subprocess. Use the deployment container for OS-level filesystem/network isolation."]
        (self.directory / "report.md").write_text(self.journal.redact("\n".join(rows) + "\n"), encoding="utf-8")
        self.journal.append("run_completed", report=report)
        self.checkpoint(active=bool(self.active_trials))
        return report

def run(config):
    # Lock before reading/repairing journal or recovering any in-flight calls.
    if config.run_dir is None:
        config.run_dir = config.task_path.parent / ".automl" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    with _run_lock(config.run_dir / "run.lock"):
        return ResearchRun(config).execute()
