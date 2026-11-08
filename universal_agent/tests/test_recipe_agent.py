import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import ResearchRun, run
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult, ModelResponse, TaskSpec
from automl_agent.prompts import build_messages
from automl_agent.providers import OpenAICompatibleProvider


def task_at(path):
    path.mkdir()
    x = np.linspace(0, 8, 80)
    pd.DataFrame({'x':x,'category':['a','b']*40,'outcome':x*2+np.cos(x)}).to_csv(path/'train.csv',index=False)
    pd.DataFrame({'x':[1.,None,4.],'category':['a','b','new']}).to_csv(path/'test.csv',index=False)
    (path/'task.md').write_text('Regression. Predict outcome. RMSE. train.csv test.csv. Save submission.csv.')
    return path/'task.md'


def config(task,directory,offline=False):
    return parse_config(["--cv-folds", "1", "--family-trials", "3", str(task),'--run-dir',str(directory),'--time-budget','90','--trial-timeout','20',
                         '--max-trials','6','--max-steps','3',*(['--offline'] if offline else [])])


def test_default_recipe_tool_runs_two_real_candidates_and_preserves_pipeline(tmp_path,monkeypatch):
    seen=[]
    def complete(self,messages,tools,**kwargs):
        seen.append((messages,tools))
        if len(seen)==1:
            name='run_model_experiments'
            arguments={'experiments':[
                {'hypothesis':'Индикаторы пропусков при тех же деревьях','base_trial_id':'trial_0002',
                 'parameters':{'n_estimators':24},'preprocessing':{'add_missing_indicators':True}},
                {'hypothesis':'Снизить сложность деревьев','base_trial_id':'trial_0002',
                 'parameters':{'n_estimators':24,'max_leaf_nodes':32}}]}
        else:
            name='finish_research'; arguments={'reason':'Две независимые гипотезы измерены.'}
        return ModelResponse({'role':'assistant','tool_calls':[{'id':f'call{len(seen)}','type':'function',
            'function':{'name':name,'arguments':json.dumps(arguments)}}]},total_tokens=30)
    monkeypatch.setenv('AUTOML_PROVIDER','openai'); monkeypatch.setenv('OPENAI_MODEL','fake')
    monkeypatch.setattr(OpenAICompatibleProvider,'complete',complete)
    directory=tmp_path/'run'
    report=run(config(task_at(tmp_path/'data'),directory))
    assert report['llm_trials']==2,'declarative model proposals must actually execute'
    advertised={t['function']['name'] for t in seen[0][1]}
    assert advertised=={'run_model_experiments','finish_research'}
    candidates=[t for t in report['trials'] if not t['is_baseline']]
    assert all(t['recipe']['parameters']['n_estimators']==24 for t in candidates)
    assert len({t['recipe_hash'] for t in candidates})==2
    assert candidates[0]['recipe']['preprocessing']['add_missing_indicators'] is True
    processes=[json.loads((Path(t['artifact_dir'])/'process.json').read_text()) for t in candidates]
    assert all(p['n_jobs']==4 for p in processes)
    assert max(p['started_at'] for p in processes)<min(p['finished_at'] for p in processes)
    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event']=='trial_completed'])==5
    for t in candidates:
        intent=next(e['payload'] for e in events if e['event']=='trial_started' and e['payload']['trial_id']==t['trial_id'])
        assert intent['recipe']==t['recipe'] and intent['recipe_hash']==t['recipe_hash']
        assert (Path(t['artifact_dir'])/'recipe.json').is_file()
    assert report['status']=='completed' and report['validation']['valid']


def sample_task():
    return TaskSpec('task.md','train.csv','test.csv','y','classification','macro_f1',True,
                    ['x','cat'],[],['cat'],['y'],'hash',classes=['a','b'])


def test_parameter_patch_inherits_measured_auto_lr_and_rejects_ineffective_weights(monkeypatch):
    research=ResearchRun.__new__(ResearchRun)
    research.task=sample_task()
    research.config=SimpleNamespace(n_jobs=4,parallel_trials=2)
    base=ExperimentResult('trial_0001','ok','.','code','baseline',metric=.5,
                          model_details={'parameters':{'learning_rate':.48828,'iterations':150,'depth':6,'l2_leaf_reg':3,'border_count':254,'auto_class_weights':'None'}})
    base.recipe={'family':'CatBoost','parameters':{'iterations':150},'preprocessing':{}}
    base.recipe_hash='original'
    research.results=[base]; research.active_trials={}
    research.journal=SimpleNamespace(append=lambda *a,**k: {})
    captured=[]
    def evaluate(proposals,**kwargs):
        captured.extend(proposals)
        return [{'error':'stop after validated construction'} for p in proposals]
    monkeypatch.setattr(research,'experiment_batch',evaluate)
    value=research._llm_model_experiments([
        {'base_trial_id':'trial_0001','hypothesis':'Корректные веса','parameters':{'auto_class_weights':'Balanced'}},
        {'base_trial_id':'trial_0001','hypothesis':'Неверный fit-only параметр','parameters':{'sample_weight':[1,2]}}])
    assert len(captured)==1,'bad constructor weights must be rejected before worker dispatch'
    assert captured[0]['recipe']['parameters']['learning_rate']==pytest.approx(.48828)
    assert captured[0]['recipe']['parameters']['auto_class_weights']=='Balanced'
    assert 'sample_weight' in value['trials'][1]['error']
    assert 'auto_class_weights' in value['trials'][1]['error']


def test_successful_recipe_duplicates_are_rejected_without_refitting(monkeypatch):
    research=ResearchRun.__new__(ResearchRun)
    research.task=sample_task(); research.config=SimpleNamespace(n_jobs=4,parallel_trials=2)
    base=ExperimentResult('trial_0001','ok','.','code','baseline',metric=.5)
    base.recipe={'family':'CatBoost','parameters':{'iterations':150},'preprocessing':{}}
    base.recipe_hash=None
    research.results=[base]; research.active_trials={}
    research.journal=SimpleNamespace(append=lambda *a,**k: {})
    calls=[]
    monkeypatch.setattr(research,'experiment_batch',lambda p,**k:calls.append(p) or [])
    value=research._llm_model_experiments([{'base_trial_id':'trial_0001','hypothesis':'Повтор под другой формулировкой','parameters':{'iterations':150}}])
    assert not calls
    assert 'duplicate' in value['trials'][0]['error'].lower()


def test_compaction_keeps_model_facts_and_clears_stale_failure(tmp_path):
    task=sample_task()
    bad=ExperimentResult('trial_0001','invalid_code',str(tmp_path/'bad'),'bad','broken',error='unmatched bracket')
    ok=ExperimentResult('trial_0002','ok',str(tmp_path/'ok'),'ok','working',metric=.5,
                        model_details={'parameters':{'learning_rate':.48828,'auto_class_weights':'None'}})
    ok.recipe={'family':'CatBoost','parameters':{'iterations':150},'preprocessing':{}}
    for trial,code in [(bad,'broken code'),(ok,'def build_model(context):\n    return None\n')]:
        Path(trial.artifact_dir).mkdir();(Path(trial.artifact_dir)/'code.py').write_text(code)
    profile={'train_shape':[100,2],'test_shape':[5,2],'features':{f'x{i}':{'stats':'x'*1000} for i in range(50)},
             'target':{'class_counts':{'a':90,'b':10}},'available_packages':{'catboost':'1.2.8'},
             'family_comparison':{'trials':[{'family':'CatBoost','trial_id':'trial_0002','status':'ok','metric':.5,'seconds':1.,'model_details':ok.model_details}]}}
    messages=build_messages(task,profile,[bad,ok],ok,{'n_jobs':4,'experiment_mode':'recipe'},context_chars=5000,compact=True)
    content=json.loads(messages[1]['content'])
    assert content['data_profile']['target']['class_counts']=={'a':90,'b':10}
    assert content['data_profile']['available_packages']['catboost']=='1.2.8'
    assert content['last_failed_code']=='','a resolved prior failure must not masquerade as the current outcome'
    assert content['experiments'][-1]['recipe']==ok.recipe
    assert content['family_comparison']['trials'][0]['model_details']['parameters']['learning_rate']==pytest.approx(.48828)


def test_unknown_recipe_intent_is_preserved_on_recovery(tmp_path):
    from automl_agent.journal import EventJournal,atomic_write_json
    task=task_at(tmp_path/'data');directory=tmp_path/'run'
    run(config(task,directory,offline=True))
    state=json.loads((directory/'state.json').read_text());state.update(active=True,checkpoint_epoch=time.time())
    atomic_write_json(directory/'state.json',state)
    recipe={'family':'ExtraTrees','parameters':{'n_estimators':27},'preprocessing':{'add_missing_indicators':True}}
    EventJournal(directory/'events.jsonl').append('trial_started',trial_id='trial_0004',code_hash='unknown',hypothesis='unknown',origin='llm',recipe=recipe,recipe_hash='saved-hash')
    cfg=config(task,directory,offline=True);cfg.resume=True
    research=ResearchRun(cfg)
    recovered=next(t for t in research.results if t.trial_id=='trial_0004')
    assert recovered.status=='interrupted'
    assert getattr(recovered,'recipe',None)==recipe and recovered.recipe_hash=='saved-hash'


def test_audit_shows_applied_recipe_inheritance_and_rejections(tmp_path):
    from automl_agent.audit import AuditJournal
    journal=AuditJournal(tmp_path/'events.jsonl',echo=False)
    recipe={'family':'CatBoost','parameters':{'learning_rate':.48828,'auto_class_weights':'Balanced'},'preprocessing':{}}
    journal.append('trial_started',trial_id='trial_0004',hypothesis='Веса при прежнем LR',recipe=recipe,parent_trial_id='trial_0001')
    journal.append('recipe_rejected',base_trial_id='trial_0001',hypothesis='Неверный параметр',error='Use auto_class_weights instead of sample_weight')
    content=(tmp_path/'agent.log').read_text()
    assert '0.48828' in content and 'Balanced' in content and 'trial_0001' in content
    assert 'sample_weight' in content
    decisions=[json.loads(s) for s in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert decisions[0]['details']['recipe']==recipe


def test_new_requests_cannot_bypass_recipe_mode_with_hidden_code_tool():
    from automl_agent.tools import ToolRegistry
    registry=ToolRegistry();called=[]
    registry.register('legacy_code','old interface',{'type':'object'},lambda **k:called.append(True) or 'recovered',advertised=False)
    result=registry.execute('legacy_code',{})
    assert result['ok'] is False and not called
    assert registry.execute('legacy_code',{},allow_hidden=True)=={'ok':True,'value':'recovered'}


def test_json_only_recipe_array_selects_structured_tool():
    from automl_agent.tools import parse_response_calls
    response={'content':json.dumps({'experiments':[{'hypothesis':'Поправить один параметр','base_trial_id':'trial_0002','parameters':{'n_estimators':240}}]})}
    calls=parse_response_calls(response)
    assert calls[0]['function']['name']=='run_model_experiments'


def test_classification_feedback_contains_host_class_diagnostics(tmp_path):
    data=tmp_path/'data';data.mkdir()
    x=np.arange(80)
    pd.DataFrame({'x':x,'cat':['a','b']*40,'label':np.where(x<64,'major','rare')}).to_csv(data/'train.csv',index=False)
    pd.DataFrame({'x':[1,70],'cat':['a','new']}).to_csv(data/'test.csv',index=False)
    (data/'task.md').write_text('Classification. Predict label. Metric macro_f1. train.csv test.csv. Save submission.csv.')
    cfg=config(data/'task.md',tmp_path/'run',offline=True);cfg.max_trials=3
    report=run(cfg)
    for trial in report['trials']:
        assert trial['diagnostics']['total_classes']==2
        assert sum(row['support'] for row in trial['diagnostics']['per_class'])==16
        assert {row['label'] for row in trial['diagnostics']['per_class']}=={'major','rare'}
        assert all(0<=row['recall']<=1 and 0<=row['f1']<=1 for row in trial['diagnostics']['per_class'])


def test_regression_recipe_prompt_has_no_class_balancing_priority():
    task=sample_task();task.task_type='regression';task.metric='rmse';task.classes=[]
    messages=build_messages(task,{'features':{}},[],None,{'n_jobs':4,'experiment_mode':'recipe'})
    assert messages[0]['content'].startswith('This task is REGRESSION')
    assert 'For imbalanced classification and macro_f1' not in messages[0]['content']
    content=json.loads(messages[1]['content'])
    assert 'NOT target classes' in content['task_constraints']
    assert 'median imputation' in content['pipeline_contract']['ExtraTrees/LightGBM/XGBoost']
