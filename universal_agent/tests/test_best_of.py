"""Behavior required by the best-of integration; no external LLM needed."""
import json
import numpy as np
import pandas as pd
import pytest

from automl_agent.config import parse_config
from automl_agent.recipes import recipe_code, recipe_fingerprint
from automl_agent.tools import validate_candidate_code


def build(ops, family='ExtraTrees'):
    recipe = {'family': family, 'parameters': {'n_estimators': 10} if family != 'CatBoost' else {'iterations': 10},
              'preprocessing': {'feature_ops': ops}}
    code = recipe_code(recipe, 'regression', n_jobs=1)
    validate_candidate_code(code)
    scope = {}
    exec(code, scope)
    return scope['build_model']({'task_type': 'regression', 'feature_columns': ['cat','x','z'],
                                'categorical_columns': ['cat'], 'n_jobs': 1, 'seed': 42})


def test_recipe_features_fit_only_training_statistics_and_keep_schema():
    model = build([{'op':'freq_encode','cols':['cat']},
                   {'op':'group_stats','by':['cat'],'cols':['x'],'stats':['mean']},
                   {'op':'bin','cols':['x'],'q':2}])
    train = pd.DataFrame({'cat':['a','a','b','b'], 'x':[0.,2.,4.,6.], 'z':[1,2,3,4]})
    model.fit(train, [0.,1.,2.,3.])
    transform = model.named_steps['features']
    probe = pd.DataFrame({'cat':['a','new'], 'x':[100.,200.], 'z':[1,1]})
    a = transform.transform(probe)
    b = transform.transform(pd.concat([probe, probe.iloc[[0]]], ignore_index=True)).iloc[:2]
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, '__ua_0_cat_freq'] == .5
    assert a.loc[1, '__ua_0_cat_freq'] == 0
    assert a.loc[0, '__ua_1_x_mean'] == 1
    assert pd.isna(a.loc[1, '__ua_1_x_mean'])
    assert list(a.columns) == list(transform.transform(train).columns)
    assert np.isfinite(model.predict(probe)).all()


@pytest.mark.parametrize('family',['ExtraTrees','CatBoost','LightGBM','XGBoost','Linear'])
def test_features_work_for_every_model_family_and_serialization(family):
    import cloudpickle
    ops=[{'op':'log1p','cols':['x']},{'op':'missing_flags','cols':['z']},
         {'op':'ratio','pairs':[['x','z']]},{'op':'combine','cols':['cat','z']}]
    model=build(ops, family) if family in ['ExtraTrees','CatBoost'] else None
    if model is None:
        code=recipe_code({'family':family,'preprocessing':{'feature_ops':ops}},'regression',1)
        scope={}; exec(code,scope)
        model=scope['build_model']({'task_type':'regression','feature_columns':['cat','x','z'],
                                   'categorical_columns':['cat'],'n_jobs':1,'seed':42})
    frame=pd.DataFrame({'cat':['a','b']*20,'x':np.arange(40.),'z':[0.,np.nan,2.,3.]*10})
    model.fit(frame,np.arange(40.))
    before=model.predict(frame)
    np.testing.assert_allclose(cloudpickle.loads(cloudpickle.dumps(model)).predict(frame), before)


def test_feature_ops_participate_in_deduplication():
    base={'family':'ExtraTrees'}
    changed={**base,'preprocessing':{'feature_ops':[{'op':'log1p','cols':['x']}]}}
    assert recipe_fingerprint(base,'regression') != recipe_fingerprint(changed,'regression')


def test_default_provider_is_local_ollama_without_key(monkeypatch,tmp_path):
    for key in ['AUTOML_PROVIDER','OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','OLLAMA_MODEL']:
        monkeypatch.delenv(key,raising=False)
    config=parse_config([str(tmp_path/'task.md')])
    assert config.provider == 'ollama'
    assert config.model == 'qwen3.5:9b'
    assert config.api_key == ''


def test_explicit_platform_endpoint_keeps_openai_interface(monkeypatch,tmp_path):
    monkeypatch.delenv('AUTOML_PROVIDER',raising=False)
    monkeypatch.setenv('OPENAI_BASE_URL','https://example.test/v1')
    monkeypatch.setenv('OPENAI_MODEL','platform-model')
    assert parse_config([str(tmp_path/'task.md')]).provider == 'openai'


def test_fallback_exists_before_expensive_profile(tmp_path,monkeypatch):
    from automl_agent.agent import ResearchRun
    from automl_agent.config import Config
    (tmp_path/'task.md').write_text('Regression. Metric: RMSE. train.csv test.csv')
    pd.DataFrame({'x':range(10),'target':range(10)}).to_csv(tmp_path/'train.csv',index=False)
    pd.DataFrame({'x':[2,3]}).to_csv(tmp_path/'test.csv',index=False)
    config=Config(tmp_path/'task.md','ollama','http://127.0.0.1:11434','qwen3.5:9b','',
                  run_dir=tmp_path/'run',offline=True)
    run=ResearchRun(config)
    def profile(task):
        assert (tmp_path/'submission.csv').exists(), 'Fallback must precede expensive diagnostics'
        raise RuntimeError('probe stops before diagnostics')
    monkeypatch.setattr('automl_agent.data.profile_task',profile)
    with pytest.raises(RuntimeError,match='probe stops'):
        run.execute()


def test_duplicate_feature_rows_stay_in_same_cv_fold():
    from automl_agent.validation import make_validation_plan
    from automl_agent.contracts import TaskSpec
    frame=pd.DataFrame({'x':np.repeat(np.arange(20),3), 'label':np.repeat([0,1]*10,3)})
    task=TaskSpec('','','','label','classification','macro_f1',True,['x'],[],[],['label'],'hash',classes=[0,1])
    plan=make_validation_plan(task,frame,pd.DataFrame({'x':[21]}),folds=3)
    assert plan['strategy'] == 'stratified_group'
    for fold in plan['folds']:
        assert set(frame.x.iloc[fold['train']]).isdisjoint(frame.x.iloc[fold['validation']])


def test_text_vocabulary_fits_train_and_survives_unknown_words():
    model=build([{'op':'text_stats','cols':['cat'],'top_k':4},{'op':'drop','cols':['cat']}])
    frame=pd.DataFrame({'cat':['red good red','blue bad blue']*10,'x':range(20),'z':range(20)})
    model.fit(frame,np.arange(20.))
    transform=model.named_steps['features']
    probe=frame.iloc[:1].assign(cat='brandnew unseen words')
    before=transform.transform(probe)
    assert before['__ua_0_cat_chars'].iloc[0] == 21
    assert before['__ua_0_cat_words'].iloc[0] == 3
    assert before.filter(like='_token_').to_numpy().sum() == 0
    np.testing.assert_array_equal(before,transform.transform(probe))


def test_numeric_feature_operations_reject_non_numeric_category():
    model=build([{'op':'log1p','cols':['cat']}])
    frame=pd.DataFrame({'cat':['a','b'],'x':[1.,2.],'z':[2.,3.]})
    with pytest.raises(ValueError,match='numeric'):
        model.fit(frame,[0.,1.])


def test_invalid_feature_schema_always_returns_value_error():
    from automl_agent.features import validate_feature_ops
    for operation in [{'op':{}},{'op':'group_stats','by':['cat'],'cols':['x'],'stats':[{}]},
                      {'op':'drop','cols':[{}]}]:
        with pytest.raises(ValueError):
            validate_feature_ops([operation])


def test_feature_hypothesis_runs_through_llm_tool_cv_and_final_refit(tmp_path,monkeypatch):
    import joblib
    from automl_agent.agent import run
    from automl_agent.contracts import ModelResponse
    from automl_agent.providers import OllamaProvider
    from automl_agent.recipes import default_recipe
    frame=pd.DataFrame({'id':range(90),'cat':['a','b','c']*30,'x':np.arange(90.),'y':np.sin(np.arange(90.)/5)})
    frame.to_csv(tmp_path/'train.csv',index=False)
    frame.drop(columns='y').iloc[:4].to_csv(tmp_path/'test.csv',index=False)
    pd.DataFrame({'id':range(4),'y':0}).to_csv(tmp_path/'sample_submission.csv',index=False)
    (tmp_path/'task.md').write_text('Regression. Primary metric: RMSE. train.csv test.csv sample_submission.csv')
    recipe=default_recipe('ExtraTrees','regression');recipe['parameters']['n_estimators']=10
    monkeypatch.setattr('automl_agent.recipes.initial_research_recipes',lambda *a:[recipe])
    def complete(*args,**kwargs):
        call={'experiments':[{'hypothesis':'Проверить train-only частоты и логарифм.',
              'base_trial_id':'trial_0001','parameters':{'min_samples_leaf':1},
              'preprocessing':{'feature_ops':[{'op':'freq_encode','cols':['cat']},{'op':'log1p','cols':['x']}]}}]}
        return ModelResponse({'role':'assistant','tool_calls':[{'id':'feature','type':'function',
                              'function':{'name':'run_model_experiments','arguments':json.dumps(call)}}]},total_tokens=20)
    monkeypatch.setattr(OllamaProvider,'complete',complete)
    monkeypatch.setenv('AUTOML_PROVIDER','ollama')
    config=parse_config([str(tmp_path/'task.md'),'--run-dir',str(tmp_path/'run'),'--time-budget','60',
                         '--cv-folds','3','--family-trials','1','--max-trials','2','--max-steps','1','--n-jobs','1'])
    report=run(config)
    trial=next(t for t in report['trials'] if t['origin']=='llm')
    assert trial['status']=='ok' and report['llm_trials']==1
    saved=joblib.load(trial['model_path'])
    probe=frame.iloc[:3][['cat','x']]
    assert all('features' in m.named_steps for m in saved.models)
    assert np.isfinite(saved.predict(probe)).all()
    assert report['final_fit']['status']=='ok' and report['validation']['valid']


def test_search_flag_evaluates_candidates_without_counting_them_as_llm(tmp_path,monkeypatch):
    from automl_agent.agent import run
    from automl_agent.recipes import default_recipe
    frame=pd.DataFrame({'x':range(60),'target':np.sin(np.arange(60))})
    frame.to_csv(tmp_path/'train.csv',index=False)
    frame[['x']].iloc[:3].to_csv(tmp_path/'test.csv',index=False)
    (tmp_path/'task.md').write_text('Regression. Metric: RMSE. train.csv test.csv')
    monkeypatch.setattr('automl_agent.recipes.initial_research_recipes',lambda *a:[default_recipe('ExtraTrees','regression')])
    report=run(parse_config([str(tmp_path/'task.md'),'--offline','--search-trials','2','--family-trials','1',
                             '--max-trials','3','--time-budget','60','--cv-folds','2','--n-jobs','1',
                             '--run-dir',str(tmp_path/'run')]))
    assert len(report['trials']) == 3
    assert sum(r['origin']=='search' and r['status']=='ok' for r in report['trials']) == 2
    assert report['llm_trials'] == 0 and report['llm_calls'] == 0


def test_cli_deadline_covers_host_work_and_retains_valid_submission(tmp_path):
    import subprocess,sys,time
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'train.csv').write_text('x,y\n1,1\n2,2\n3,3\n')
    (tmp_path/'test.csv').write_text('x\n4\n')
    task=tmp_path/'task.md';task.write_text('Regression. RMSE. train.csv test.csv')
    code='''import sys,time,json
sys.path.insert(0,sys.argv[1])
import solution
import automl_agent.agent
from pathlib import Path
def slow(config):
    (config.task_path.parent/'submission.csv').write_text('y\\n2.0\\n')
    time.sleep(10)
    return {}
automl_agent.agent.run=slow
raise SystemExit(solution.main([sys.argv[2],'--offline','--time-budget','2']))
'''
    started=time.monotonic()
    completed=subprocess.run([sys.executable,'-c',code,str(root),str(task)],capture_output=True,text=True,timeout=6)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['status'] == 'deadline'
    assert time.monotonic()-started < 4
