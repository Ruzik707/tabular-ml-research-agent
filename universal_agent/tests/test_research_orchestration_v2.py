"""Budget and LLM behavior checked through the real host orchestration."""
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import ResearchRun
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult, ModelResponse, ProcessResult
from automl_agent.recipes import default_recipe


def host(tmp_path):
    data = tmp_path / 'data'; data.mkdir()
    pd.DataFrame({'id':range(60), 'x':np.arange(60), 'y':np.arange(60)*2.}).to_csv(data/'train.csv',index=False)
    pd.DataFrame({'id':[100,101], 'x':[2.,3.]}).to_csv(data/'test.csv',index=False)
    pd.DataFrame({'id':[100,101], 'y':[0.,0.]}).to_csv(data/'sample_submission.csv',index=False)
    (data/'task.md').write_text('Regression. Predict y. RMSE. train.csv test.csv sample_submission.csv.')
    cfg=parse_config([str(data/'task.md'),'--offline','--run-dir',str(tmp_path/'run'),
        '--time-budget','180','--trial-timeout','120','--parallel-trials','1','--n-jobs','1',
        '--cv-folds','1','--max-steps','1','--max-trials','12'])
    cfg.offline=False; cfg.provider='ollama'; cfg.model='fake'
    research=ResearchRun(cfg)
    research.profile={'train_shape':[60,1],'test_shape':[2,1],'available_packages':{'lightgbm':'4.6.0'}}
    research.split={k:str(tmp_path/(k+'.csv')) for k in ['train_features','train_target','validation_features','test_features','full_train_features','full_train_target']}
    research.split.update(split_fingerprint='fixture',mode='evaluate')
    folder=research.directory/'trial_0001';folder.mkdir()
    source='from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n'
    (folder/'code.py').write_text(source)
    base=ExperimentResult('trial_0001','ok',str(folder),hashlib.sha256(source.encode()).hexdigest(),'measured baseline',
        metric=2.,metrics={'rmse':2.},seconds=2.,is_baseline=True,origin='family',
        recipe=default_recipe('LightGBM','regression'),model_details={'training':{'estimated_refit_seconds':3.}})
    research.results=[base]
    research._update_reserve(record=False)
    return research


def test_two_llm_proposals_are_accepted_with_one_worker(tmp_path,monkeypatch):
    research=host(tmp_path)
    captured=[]
    monkeypatch.setattr(research,'experiment_batch',lambda rows,**kwargs: captured.extend(rows) or [{'status':'queued'} for _ in rows])
    result=research._llm_model_experiments([
        {'base_trial_id':'trial_0001','hypothesis':'fewer leaves','parameters':{'num_leaves':15}},
        {'base_trial_id':'trial_0001','hypothesis':'more leaves','parameters':{'num_leaves':47}},
    ])
    assert len(captured)==2
    assert len(result['trials'])==2
    assert len({r['recipe_hash'] for r in captured})==2
    assert research.config.parallel_trials==1


def test_single_trial_cannot_consume_the_whole_short_research_budget(tmp_path):
    research=host(tmp_path)
    _,job=research._prepare_trial('bounded candidate',
        'from sklearn.dummy import DummyRegressor\ndef build_model(context):\n return DummyRegressor(strategy="median")\n')
    assert 0 < job['timeout'] <= 30
    assert job['timeout'] < research.budget.research_remaining()/2


def test_final_fit_is_not_started_when_its_complete_window_does_not_fit(tmp_path,monkeypatch):
    research=host(tmp_path)
    research.reserve_plan.update(estimated_fit_seconds=5.,startup_guard_seconds=8.,overhead_seconds=5.,required_seconds=18.,feasible=True)
    monkeypatch.setattr(research.budget,'remaining',lambda:10.)
    calls=[]
    def process(*args,**kwargs):
        calls.append(kwargs)
        return ProcessResult('timeout',-15,'','',0.,0.)
    monkeypatch.setattr('automl_agent.runtime.run_process',process)
    outcome=research.finalize(research.results[0])
    assert not calls
    assert outcome['status']=='skipped'
    assert 'window' in outcome['reason'].lower()


def test_local_search_progresses_while_llm_response_is_pending(tmp_path,monkeypatch):
    research=host(tmp_path)
    searched=threading.Event()
    def local(limit):
        searched.set()
        research.results.append(replace(research.results[0],trial_id='trial_0002',is_baseline=False,origin='search'))
    monkeypatch.setattr(research,'bounded_search',local)
    def complete(self,*args,**kwargs):
        searched.wait(.5)
        return ModelResponse({'role':'assistant','tool_calls':[{'id':'finish','type':'function',
            'function':{'name':'finish_research','arguments':json.dumps({'reason':'measured local search complete'})}}]},total_tokens=12)
    monkeypatch.setattr('automl_agent.providers.OllamaProvider.complete',complete)
    research.research()
    assert searched.is_set(),'local HPO must make progress before the slow LLM returns'
    assert research.tokens_used==12
    assert research.finish_requested
    events=research.journal.read_events()
    assert len([e for e in events if e['event']=='llm_response'])==1
    assert len([e for e in events if e['event']=='llm_response_consumed'])==1


def test_accepted_ensemble_must_fit_its_own_window(tmp_path,monkeypatch):
    research=host(tmp_path)
    research.reserve_plan.update(estimated_fit_seconds=5.,startup_guard_seconds=8.,overhead_seconds=5.,required_seconds=18.,feasible=True)
    research.postprocessing_plan={'accepted':True,'estimated_fit_seconds':40.,
        'member_trial_ids':['trial_0001'],'weights':[1.],'class_multipliers':None}
    monkeypatch.setattr(research.budget,'remaining',lambda:50.)
    calls=[]
    monkeypatch.setattr('automl_agent.runtime.run_process',lambda *a,**kw: calls.append(kw) or ProcessResult('timeout',-15,'','',0.,0.))
    outcome=research.finalize(research.results[0])
    assert not calls,'single-model feasibility cannot authorize a more costly accepted ensemble'
    assert outcome['status']=='skipped'


def test_code_proposal_batch_is_independent_of_worker_concurrency(tmp_path,monkeypatch):
    research=host(tmp_path)
    monkeypatch.setattr(research,'experiment_batch',lambda rows: [replace(research.results[0],trial_id=f'trial_{i+2:04d}') for i,_ in enumerate(rows)])
    outcome=research._llm_experiments([{'hypothesis':'first'},{'hypothesis':'second'}])
    assert len(outcome['trials'])==2


def test_postprocessing_rejects_ensemble_that_leaves_no_startup_guard(tmp_path,monkeypatch):
    import inspect
    from automl_agent.ensemble import choose_postprocessing_bounded
    research=host(tmp_path)
    def process(*args,**kwargs):
        manifest=json.loads((Path(kwargs['cwd'])/'manifest.json').read_text())
        Path(manifest['result_path']).write_text(json.dumps({'accepted':True,'estimated_fit_seconds':40.}))
        return ProcessResult('ok',0,'','',.1,0.)
    monkeypatch.setattr('automl_agent.runtime.run_process',process)
    startup={'startup_guard_seconds':8.} if 'startup_guard_seconds' in inspect.signature(choose_postprocessing_bounded).parameters else {}
    plan=choose_postprocessing_bounded(research.task,research.split,research.results,research.directory,
        remaining_seconds=50.,refit_seconds=5.,overhead_seconds=5.,**startup)
    assert plan['accepted'] is False,'40s fit + 8s startup + 5s write cannot fit in 50s'
    assert plan['reason']=='insufficient_refit_time'


def test_affordable_ensemble_replaces_an_infeasible_single_plan(tmp_path,monkeypatch):
    research=host(tmp_path)
    research.reserve_plan.update(feasible=False,estimated_fit_seconds=200.,required_seconds=213.,startup_guard_seconds=8.,overhead_seconds=5.)
    research.postprocessing_plan={'accepted':True,'estimated_fit_seconds':5.,'member_trial_ids':['trial_0001'],
        'weights':[1.],'class_multipliers':None}
    monkeypatch.setattr(research.budget,'remaining',lambda:50.)
    calls=[]
    monkeypatch.setattr('automl_agent.runtime.run_process',lambda *a,**kw: calls.append(kw) or ProcessResult('timeout',-15,'','',0.,0.))
    research.finalize(research.results[0])
    assert len(calls)==1,'feasibility must be recomputed for the actual accepted final members'
    assert research.reserve_plan['feasible'] is True
    assert research.reserve_plan['required_seconds']==18.


@pytest.mark.parametrize('has_candidate',[True,False])
def test_prompt_context_failure_still_reaches_fallback_and_finalization(tmp_path,monkeypatch,has_candidate):
    research=host(tmp_path)
    if not has_candidate:
        research.results=[]
    monkeypatch.setattr(research,'_family_comparison',lambda strategy:None)
    def too_large(*args,**kwargs):
        raise ValueError('Recipe prompt essentials require 9150 characters, above the 9000-character budget.')
    monkeypatch.setattr('automl_agent.prompts.build_messages',too_large)
    order=[]
    monkeypatch.setattr(research,'bounded_search',lambda limit:order.append(('fallback',limit)))
    def finalize(best):
        order.append(('finalize',best.trial_id if best else None))
        if best is None:
            return {'status':'skipped','reason':'No measured candidate'}
        pd.DataFrame({'id':[100,101],'y':[4.,6.]}).to_csv(research.output,index=False)
        return {'status':'ok'}
    monkeypatch.setattr(research,'finalize',finalize)
    report=research.execute()
    assert report['stop_reason']=='prompt_context_error'
    assert report['validation']['valid']
    assert report['llm_calls']==0 and report['total_tokens']==0
    assert order==[('fallback',4),('finalize','trial_0001' if has_candidate else None)]
    assert report['final_fit']['status']==('ok' if has_candidate else 'skipped')
    events=research.journal.read_events()
    assert len([e for e in events if e['event']=='llm_context_error'])==1
    assert not any(e['event'] in {'llm_request','run_error'} for e in events)


def test_prompt_keyboard_interrupt_is_not_swallowed_as_a_context_error(tmp_path,monkeypatch):
    research=host(tmp_path)
    monkeypatch.setattr(research,'_family_comparison',lambda strategy:None)
    def interrupted(*args,**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr('automl_agent.prompts.build_messages',interrupted)
    with pytest.raises(KeyboardInterrupt):
        research.execute()
    assert not any(e['event']=='llm_context_error' for e in research.journal.read_events())
