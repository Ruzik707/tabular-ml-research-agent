"""Reproducible, externally scored comparison of pinned hackathon agents.

Development tool only. Truth stays outside task directories. Agent algorithms
are extracted unchanged from Git; a local proxy normalizes Ollama settings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import psutil


PINS = {
    "universal": ("1dd19b50188dfd64047bcf5a73339bd87bfde9c5", "universal_agent", "automl_agent"),
    "aprelskiy": ("6887598613e97c03e54f3967661a8607a9a7c417", "", "automl_agent"),
    "oleg": ("05f5c78d464d5c637020db896789a2b81c893d7b", "", "agent"),
    "rakovskiy": ("ba0f29de1e4ab5586aed21e4d2348b5b4f718181", "", "agent"),
    "ruzik": ("85f4988ba11c70ebe2242c8722b32f2d4bea5c7a", "", "ruzik_agent"),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest_files(root, files=None):
    root = Path(root)
    files = sorted(files or [p.relative_to(root).as_posix() for p in root.rglob("*.py")])
    digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    checksum = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return checksum, digests


def stage_sources(repo, directory, names, *, pins=None):
    """Extract runtime code only; never extract datasets, .env or build outputs."""
    agents = []
    for name in names:
        revision, prefix, package = (PINS if pins is None else pins)[name]
        prefix = prefix + "/" if prefix else ""
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=repo, text=True).splitlines()
        paths = [p for p in paths if p in {prefix + "solution.py", prefix + "requirements.txt", prefix + "requirements-mvp.txt"} or (
            p.startswith(prefix + package + "/") and p.endswith(".py"))]
        if prefix + "solution.py" not in paths:
            raise ValueError(f"Missing pinned entrypoint: {name}")
        raw = subprocess.check_output(["git", "archive", "--format=tar", revision, "--", *paths], cwd=repo)
        target = Path(directory) / name
        if target.is_symlink():
            raise ValueError("Pinned source directory must not be a symlink")
        # A resumed workspace must contain exactly the pinned runtime. Otherwise
        # stale modules can affect imports without appearing in its source hash.
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                if member.name not in paths or member.issym() or ".." in Path(member.name).parts:
                    raise ValueError("Unexpected archive entry")
                relative = member.name[len(prefix):]
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.extractfile(member).read())
        checksum, files = digest_files(target, [path[len(prefix):] for path in paths])
        agents.append({"kind": name, "revision": revision, "root": str(target.resolve()),
                       "source_sha256": checksum, "runtime_files": files})
    return agents


def agent_command(agent, task, run_dir, budget, threads, seed, offline=False):
    argv = [sys.executable, str(Path(agent["root"]) / "solution.py"), str(task),
            "--time-budget", str(budget), "--seed", str(seed), "--run-dir", str(run_dir)]
    kind = agent["kind"]
    if kind in {"universal", "aprelskiy"}:
        argv += ["--n-jobs", str(threads), "--parallel-trials", "1", "--cv-folds", "2",
                 "--response-tokens", "2000", "--llm-timeout", "60", "--trial-timeout", str(min(120, budget))]
    elif kind in {"oleg", "rakovskiy"}:
        argv += ["--n-jobs", str(threads), "--folds", "2"]
    elif kind == "ruzik":
        argv += ["--threads", str(threads), "--folds", "2", "--llm-timeout", "60",
                 "--trial-timeout", str(min(120, budget)), "--max-tokens", "0"]
    else:
        raise ValueError("Unknown agent kind")
    if offline:
        argv.append("--offline")
    return argv


def stop_tree(process, observed=(), force=False):
    """Stop descendants even when candidate workers create separate sessions."""
    processes = {p.pid: p for p in observed}
    try:
        parent = psutil.Process(process.pid)
        processes.update({p.pid: p for p in parent.children(recursive=True)})
        processes[parent.pid] = parent
    except psutil.Error:
        pass
    for child in reversed(list(processes.values())):
        try:
            child.kill() if force else child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(processes.values()), timeout=1.)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        process.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.)


def capture_process(argv, cwd, env, submission, snapshot, budget, logs):
    logs, submission, snapshot = Path(logs), Path(submission), Path(snapshot)
    logs.mkdir(parents=True, exist_ok=True)
    snapshot.unlink(missing_ok=True)
    observed, peak, timed_out = {}, 0, False
    with (logs / "stdout.txt").open("wb") as stdout, (logs / "stderr.txt").open("wb") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= budget:
                    timed_out = True
                    break
                try:
                    tree = [psutil.Process(process.pid)]
                    tree += tree[0].children(recursive=True)
                    observed.update({p.pid: p for p in tree})
                    rss = 0
                    for item in tree:
                        try:
                            rss += item.memory_info().rss
                        except psutil.Error:
                            pass
                    peak = max(peak, rss)
                except psutil.Error:
                    pass
                time.sleep(min(.1, max(.001, budget - (time.monotonic() - started))))
            elapsed = time.monotonic() - started
            # Freeze writers before reading the deadline artifact. Workers may
            # own separate process groups, so suspend the whole observed tree.
            if timed_out:
                try:
                    parent = psutil.Process(process.pid)
                    parent.suspend()
                    observed[parent.pid] = parent
                    for _ in range(2):
                        for child in parent.children(recursive=True):
                            observed[child.pid] = child
                            try:
                                child.suspend()
                            except psutil.Error:
                                pass
                except psutil.Error:
                    pass
            if submission.is_file():
                shutil.copyfile(submission, snapshot)
        finally:
            cleanup_started = time.monotonic()
            stop_tree(process, observed.values(), force=timed_out)
    return {"timed_out": timed_out, "returncode": process.returncode,
            "snapshot_exists": snapshot.is_file(), "seconds": round(elapsed, 3),
            "cleanup_seconds": round(time.monotonic() - cleanup_started, 3),
            "peak_agent_tree_rss_mb": round(peak / 1024**2, 1)}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # A killed writer may leave a partial trailing record.
    except OSError:
        pass
    return rows


def run_stats(kind, run_dir, proxy_log):
    report = read_json(Path(run_dir) / "report.json", {})
    if kind in {"universal", "aprelskiy"}:
        trials = report.get("trials")
        if trials is None:
            trials = [e.get("payload", {}).get("result", {}) for e in read_jsonl(Path(run_dir) / "events.jsonl")
                      if e.get("event") == "trial_completed"]
        best_score = report.get("best_metric")
        llm_trials = sum(r.get("origin") == "llm" and r.get("status") == "ok" for r in trials)
    else:
        trials = report.get("experiments")
        if not isinstance(trials, list):
            trials = read_json(Path(run_dir) / "experiments.json", []) if kind == "ruzik" else read_jsonl(Path(run_dir) / "experiments.jsonl")
        best_score = (report.get("best") or {}).get("score")
        llm_trials = (report.get("llm") or {}).get("successful_experiments", 0) if kind == "ruzik" else sum(
            r.get("source") == "llm" and r.get("status") == "ok" for r in trials)
    calls = read_jsonl(proxy_log)
    final_fit = report.get("final_fit", report.get("final"))
    return {"internal_score": best_score, "reported_status": report.get("status"),
            "trials_completed": len(trials), "trials_ok": sum(r.get("status") == "ok" for r in trials),
            "trials_failed": sum(r.get("status") != "ok" for r in trials), "successful_llm_trials": llm_trials,
            "llm_requests": sum(e.get("event") == "started" for e in calls),
            "llm_responses": sum(e.get("event") == "completed" for e in calls),
            "llm_tokens": sum((e.get("prompt_eval_count", 0) or 0) + (e.get("eval_count", 0) or 0)
                              for e in calls if e.get("event") == "completed"),
            "final_fit_status": final_fit.get("status") if isinstance(final_fit, dict) else None,
            "trial_errors": [str(r.get("error"))[:500] for r in trials if r.get("error")][:8]}


def start_proxy(directory, config):
    ready = directory / "proxy_ready.json"
    ready.unlink(missing_ok=True)
    log = directory / "ollama_calls.jsonl"
    argv = [sys.executable, str(Path(__file__).with_name("benchmark_proxy.py")), "--port", "0",
            "--ready-file", str(ready), "--log", str(log), "--upstream", config.upstream,
            "--model", config.model, "--num-ctx", "16384", "--num-predict", "2000",
            "--threads", str(config.threads), "--seed", str(config.seed)]
    stream = (directory / "proxy_stderr.txt").open("wb")
    process = subprocess.Popen(argv, stdout=stream, stderr=stream, start_new_session=True)
    deadline = time.monotonic() + 10
    while not ready.is_file() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(.05)
    if not ready.is_file():
        stop_tree(process)
        stream.close()
        raise RuntimeError("Local Ollama measurement proxy failed to start")
    return process, stream, read_json(ready)["port"], log


def run_one(agent, case, workspace, config):
    from benchmark_data import score_submission
    run_id = case["name"] + "/" + agent["kind"]
    directory = workspace / "runs" / run_id
    result_file = directory / "result.json"
    budget = config.public_budget if case["name"].startswith("public-") else config.supplemental_budget
    truth_sha256 = hashlib.sha256(Path(case["truth_path"]).read_bytes()).hexdigest()
    if result_file.is_file():
        result = read_json(result_file)
        expected = {"revision": agent["revision"], "source_sha256": agent["source_sha256"],
                    "input_hashes": case["input_hashes"], "split_fingerprint": case["split_fingerprint"],
                    "truth_sha256": truth_sha256, "budget_seconds": budget}
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cached run has incompatible fingerprints or budget: {directory}")
        captured = directory / "captured_submission.csv"
        captured_sha = hashlib.sha256(captured.read_bytes()).hexdigest() if captured.is_file() else None
        if result.get("captured_submission_sha256") != captured_sha:
            raise ValueError("Cached submission checksum changed")
        rescored = score_submission(case, directory / "captured_submission.csv")
        if result.get("valid") != rescored["valid"] or result.get("score") != rescored["score"]:
            raise ValueError("Cached submission has changed or scores differently")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    task_dir, run_dir = directory / "input", directory / "artifacts"
    if task_dir.exists():
        raise RuntimeError(f"Incomplete run exists; preserve or move it before retry: {directory}")
    shutil.copytree(case["task_dir"], task_dir)
    argv = agent_command(agent, task_dir / "task.md", run_dir, budget, config.threads, config.seed, config.offline)
    proxy, stream, port, proxy_log = start_proxy(directory, config)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "OLLAMA_", "AUTOML_", "AGENT_"))}
    env.update({"OPENAI_API_KEY": "local-benchmark-placeholder", "OPENAI_MODEL": config.model,
                "OPENAI_BASE_URL": f"http://127.0.0.1:{port}" + ("/v1" if agent["kind"] == "ruzik" else ""),
                "AUTOML_PROVIDER": "ollama", "OLLAMA_NUM_CTX": "16384", "AGENT_LLM_TIMEOUT": "60",
                "AUTOML_EXTRA_BODY": '{"options":{"num_ctx":16384}}', "PYTHONUNBUFFERED": "1"})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(config.threads)
    _, input_hashes = digest_files(task_dir, case["input_hashes"].keys())
    if input_hashes != case["input_hashes"]:
        raise ValueError("Input hashes differ before run")
    write_json(directory / "command.json", {"argv": argv, "budget_seconds": budget,
               "threads": config.threads, "source_sha256": agent["source_sha256"], "input_hashes": input_hashes})
    print(json.dumps({"event": "started", "case": case["name"], "agent": agent["kind"], "budget": budget}), flush=True)
    try:
        outcome = capture_process(argv, agent["root"], env, task_dir / "submission.csv",
                                  directory / "captured_submission.csv", budget, directory)
    finally:
        stop_tree(proxy)
        stream.close()
    score = score_submission(case, directory / "captured_submission.csv")
    captured = directory / "captured_submission.csv"
    captured_sha = hashlib.sha256(captured.read_bytes()).hexdigest() if captured.is_file() else None
    _, hashes_after = digest_files(task_dir, case["input_hashes"].keys())
    stats = run_stats(agent["kind"], run_dir, proxy_log)
    result = {"case": case["name"], "agent": agent["kind"], "revision": agent["revision"],
              "source_sha256": agent["source_sha256"], "split_fingerprint": case["split_fingerprint"],
              "input_hashes": input_hashes, "inputs_unchanged": hashes_after == input_hashes,
              "truth_sha256": truth_sha256,
              "captured_submission_sha256": captured_sha,
              "metric": case["metric"], "maximize": case["maximize"], "budget_seconds": budget,
              **outcome, **score, **stats}
    if not result["inputs_unchanged"]:
        result.update(valid=False, score=None, error="Agent modified input files")
    write_json(result_file, result)
    print(json.dumps({"event": "completed", **{k: result.get(k) for k in (
        "case", "agent", "valid", "score", "seconds", "timed_out", "llm_responses", "trials_ok")}}, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--agents", nargs="+", choices=sorted(PINS), default=list(PINS))
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--public-budget", type=int, default=180)
    parser.add_argument("--supplemental-budget", type=int, default=60)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--universal-ref", help="Explicit frozen Universal revision for before/after comparisons")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    pins = dict(PINS)
    if args.universal_ref:
        if 'universal' not in args.agents:
            parser.error('--universal-ref requires universal in --agents')
        revision = subprocess.check_output(['git', 'rev-parse', '--verify', args.universal_ref + '^{commit}'],
                                           cwd=args.repo, text=True).strip()
        pins['universal'] = (revision, *PINS['universal'][1:])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_data import prepare_cases
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    config = {"seed": args.seed, "threads": args.threads, "public_budget": args.public_budget,
              "supplemental_budget": args.supplemental_budget, "model": args.model, "offline": args.offline,
              "num_ctx": 16384, "num_predict": 2000, "think": False, "temperature": .2,
              "upstream": args.upstream, "agents": args.agents, "cases": args.cases,
              "pins": {n: pins[n][0] for n in args.agents}}
    try:
        with urllib.request.urlopen(args.upstream.rstrip('/') + '/api/tags', timeout=10) as response:
            models = json.load(response).get('models', [])
        config['ollama_model_digest'] = next(m['digest'] for m in models if m.get('name') == args.model or m.get('model') == args.model)
    except (OSError, StopIteration, ValueError):
        if not args.offline:
            raise ValueError('Requested local Ollama model is unavailable') from None
        config['ollama_model_digest'] = None
    harness_files = ['compare_agents.py', 'benchmark_data.py', 'benchmark_proxy.py']
    config['harness_sha256'], config['harness_file_hashes'] = digest_files(Path(__file__).parent, harness_files)
    previous = read_json(workspace / "config.json")
    if previous is not None and previous != config:
        raise ValueError("Refusing to mix results from different benchmark configurations")
    write_json(workspace / "config.json", config)
    agents = stage_sources(args.repo, workspace / "sources", args.agents, pins=pins)
    write_json(workspace / "sources.json", agents)
    cases = prepare_cases(args.data_root, workspace / "cases", seed=args.seed)
    if args.cases:
        cases = [case for case in cases if case["name"] in args.cases]
        if len(cases) != len(args.cases):
            raise ValueError("Unknown or repeated benchmark cases")
    write_json(workspace / "cases.json", cases)
    environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                   "physical_cpus": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
                   "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1), "packages": {}}
    for package in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "pydantic", "python-dotenv", "openai", "psutil"):
        environment["packages"][package] = importlib.metadata.version(package)
    previous_env = read_json(workspace / "environment.json")
    if previous_env is not None and previous_env != environment:
        raise ValueError('Refusing resume with a different runtime environment')
    write_json(workspace / "environment.json", environment)
    outcomes = []
    for case in cases:
        # Deterministic rotation reduces systematic first/last model order.
        order = list(agents)
        random.Random(str(args.seed) + "/" + case["name"]).shuffle(order)
        for agent in order:
            if digest_files(Path(__file__).parent, harness_files)[0] != config['harness_sha256']:
                raise ValueError('Benchmark harness changed during measurement')
            outcomes.append(run_one(agent, case, workspace, args))
            write_json(workspace / "results.json", outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
