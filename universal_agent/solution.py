"""Platform entry point: python solution.py /path/to/task.md."""

from __future__ import annotations

import time
PROCESS_STARTED_AT = time.monotonic()

import json
import signal
import sys

from automl_agent.config import parse_config


class DeadlineExceeded(KeyboardInterrupt):
    """Interrupt host preparation as well as supervised training workers."""


def deadline_report(config):
    from automl_agent.contracts import TaskSpec
    from automl_agent.data import discover_task
    from automl_agent.evaluation import validate_submission
    saved = config.run_dir/'task.json' if config.run_dir else None
    task = TaskSpec.from_dict(json.loads(saved.read_text())) if saved and saved.is_file() else discover_task(
        config.task_path, overrides={k:v for k,v in {'target':config.target,'metric':config.metric,
                                                   'task_type':config.task_type}.items() if v is not None})
    output = config.task_path.parent/'submission.csv'
    validation = validate_submission(task, output)
    report = {'status':'deadline','stop_reason':'wall_clock_limit','submission_path':str(output),
              'validation':validation,'elapsed_seconds':round(time.monotonic()-PROCESS_STARTED_AT,3)}
    if config.run_dir:
        from automl_agent.journal import atomic_write_json
        atomic_write_json(config.run_dir/'deadline_report.json',report)
    return report


def main(argv=None):
    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    config = parse_config(argv)
    config.process_started_at = PROCESS_STARTED_AT
    previous_alarm = None
    expired = False
    def deadline(_signum, _frame):
        nonlocal expired
        expired = True
        raise DeadlineExceeded
    if hasattr(signal, 'setitimer'):
        previous_alarm = signal.signal(signal.SIGALRM, deadline)
        margin = min(5., max(.05, config.time_budget*.01))
        signal.setitimer(signal.ITIMER_REAL, max(.01, config.time_budget-(time.monotonic()-PROCESS_STARTED_AT)-margin))
    try:
        from automl_agent.agent import run

        report = run(config)
    except KeyboardInterrupt:
        if not expired:
            print("Interrupted; use --resume with the same --run-dir", file=sys.stderr)
            return 130
        # Never invent a successful model/refit: validate the durable checkpoint.
        try:
            report = deadline_report(config)
        except Exception as exc:
            print('Deadline reached without a valid checkpoint: '+type(exc).__name__, file=sys.stderr)
            return 1
    except Exception as exc:
        message = str(exc)
        if config.api_key:
            message = message.replace(config.api_key, "[REDACTED]")
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return 1
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
