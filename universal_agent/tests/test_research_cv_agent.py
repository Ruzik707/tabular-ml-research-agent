import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.agent import run
from automl_agent.config import parse_config
from automl_agent.contracts import ModelResponse
from automl_agent.providers import OpenAICompatibleProvider
from automl_agent.recipes import initial_research_recipes


@pytest.mark.parametrize('classification', [False, True])
def test_cv_research_fits_two_edits_and_refits_selected_configuration(tmp_path, monkeypatch, classification):
    task_dir=tmp_path/'data';task_dir.mkdir()
    frame=pd.DataFrame({'id':range(160),'x':np.arange(160)/20,'cat':['a','b']*80,
                        'y':['rare' if i%5==0 else 'common' for i in range(160)] if classification else np.arange(160)/5})
    frame.to_csv(task_dir/'train.csv',index=False)
    frame.drop(columns='y').iloc[:12].to_csv(task_dir/'test.csv',index=False)
    pd.DataFrame({'id':range(12),'y':0}).to_csv(task_dir/'sample_submission.csv',index=False)
    (task_dir/'task.md').write_text(('Classification. Macro F1.' if classification else 'Regression. RMSE.')+' train.csv test.csv sample_submission.csv. Predict y.')
    def cheap(task_type, packages):
        recipes=initial_research_recipes(task_type,packages)
        for recipe in recipes:
            if recipe['family']=='CatBoost':recipe['parameters']['iterations']=20
            elif recipe['family'] in {'LightGBM','XGBoost'}:recipe['parameters']['n_estimators']=20
            elif recipe['family']=='ExtraTrees':recipe['parameters']['n_estimators']=10
            if recipe.get('fit_options',{}).get('early_stopping_rounds'):recipe['fit_options']['early_stopping_rounds']=3
        return recipes
    monkeypatch.setattr('automl_agent.recipes.initial_research_recipes',cheap)
    seen=[]
    def complete(self,messages,tools,**kwargs):
        seen.append(messages)
        content=json.loads(messages[1]['content'])
        if len(seen)==1:
            assert content['data_profile']['validation_plan']['fold_count']==5
            assert 'dataset_diagnostics' in content['data_profile']
            base=next(r for r in content['model_catalog']['measured_bases'] if r['family']=='ExtraTrees')
            assert content['best_experiment']['diagnostics']['validation']['coverage']==1
            changes=[{} if classification else {'n_estimators':12},{'n_estimators':14}]
            args={'experiments':[{'hypothesis':'Проверить число деревьев по OOF','base_trial_id':base['trial_id'],'parameters':change,
                    **({'fit_options':{'sample_weight':'sqrt_balanced'}} if classification else {})} for change in changes]}
            name='run_model_experiments'
        else:
            name='finish_research';args={'reason':'Тестовые независимые гипотезы выполнены.'}
        return ModelResponse({'role':'assistant','tool_calls':[{'id':f'c{len(seen)}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]},total_tokens=20)
    monkeypatch.setattr(OpenAICompatibleProvider,'complete',complete)
    monkeypatch.setenv('OPENAI_MODEL','fixture');monkeypatch.setenv('AUTOML_PROVIDER','openai')
    directory=tmp_path/'run'
    args=[str(task_dir/'task.md'),'--run-dir',str(directory),'--time-budget','90','--trial-timeout','20','--max-trials','8','--max-steps','3','--n-jobs','1']
    report=run(parse_config(args))
    assert report['status']=='completed'
    assert report['llm_trials']==2
    assert report['final_fit']['status']=='ok'
    assert len(report['trials'])==6
    assert all(t['status']=='ok' for t in report['trials'])
    assert all(len(t['diagnostics']['validation']['folds'])==5 for t in report['trials'])
    if classification:
        assert all(t['recipe']['fit_options']['sample_weight']=='sqrt_balanced' for t in report['trials'] if not t['is_baseline'])
    best=next(t for t in report['trials'] if t['trial_id']==report['selected_trial'])
    final=report['final_fit']['model_details']
    if best['model_details']['training']['final_iterations']:
        assert final['training']['best_iteration']==best['model_details']['training']['final_iterations']
    loaded=joblib.load(report['final_fit']['model_path'])
    predictions=loaded.predict(pd.read_csv(task_dir/'test.csv')[['x','cat']])
    if classification:
        np.testing.assert_array_equal(predictions,pd.read_csv(task_dir/'submission.csv')['y'].to_numpy())
    else:
        np.testing.assert_allclose(predictions,pd.read_csv(task_dir/'submission.csv')['y'].to_numpy(),rtol=1e-12,atol=1e-12)
    resumed=run(parse_config([*args,'--resume']))
    assert resumed['selected_trial']==report['selected_trial'] and len(seen)==2
