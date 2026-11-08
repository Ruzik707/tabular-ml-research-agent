import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from automl_agent.contracts import ExperimentResult, TaskSpec
from automl_agent.ensemble import choose_postprocessing


def test_postprocessing_timeout_protects_the_final_single_model_refit(tmp_path, monkeypatch):
    from automl_agent.ensemble import choose_postprocessing_bounded
    calls=[]
    def timeout(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status='timeout', seconds=10., to_dict=lambda:{'status':'timeout','seconds':10.})
    monkeypatch.setattr('automl_agent.runtime.run_process', timeout)
    task=TaskSpec('task.md','train.csv','test.csv','y','regression','rmse',False,['x'],[],[],['y'],'hash')
    plan=choose_postprocessing_bounded(task,{},[],tmp_path,remaining_seconds=50.,overhead_seconds=5.,refit_seconds=35.)
    assert plan['reason']=='postprocessing_timeout' and not plan['accepted']
    assert calls[0]['timeout']==10. and calls[0]['n_jobs']==1
    calls.clear()
    skipped=choose_postprocessing_bounded(task,{},[],tmp_path,remaining_seconds=39.,overhead_seconds=5.,refit_seconds=35.)
    assert skipped['reason']=='insufficient_postprocessing_time' and not calls


def test_host_blends_only_matching_oof_and_checks_cost(tmp_path):
    y=np.arange(60)/3
    ids=np.repeat(np.arange(3),20)
    pd.DataFrame({'y':y}).to_csv(tmp_path/'valid_y.csv',index=False)
    pd.DataFrame({'id':[0,1],'x':[1,2]}).to_csv(tmp_path/'test.csv',index=False)
    spec=TaskSpec('task.md','train.csv',str(tmp_path/'test.csv'),'y','regression','rmse',False,['x'],['id'],[],['id','y'],'hash')
    trials=[]
    for i,sign in enumerate([1,-1]):
        path=tmp_path/f'trial_{i}';path.mkdir()
        pd.DataFrame({'y':y+sign}).to_csv(path/'valid_predictions.csv',index=False)
        pd.DataFrame({'y':np.array([3.,5.])+sign}).to_csv(path/'test_predictions.csv',index=False)
        np.save(path/'oof_fold_ids.npy',ids)
        trials.append(ExperimentResult(f'trial_{i:04d}','ok',str(path),'code','hypothesis',metric=1.,
                      recipe={'family':['CatBoost','ExtraTrees'][i]},
                      model_details={'training':{'split_fingerprint':'same','estimated_refit_seconds':2.}}))
    split={'mode':'evaluate_cv','split_fingerprint':'same','validation_target':str(tmp_path/'valid_y.csv'),
           'strategy':'kfold'}
    # Source scoring/alignment is already host verified at commit; choose rechecks trial fingerprint and fold axes.
    outcome=choose_postprocessing(spec,split,trials,tmp_path,remaining_seconds=100,overhead_seconds=5)
    assert outcome['accepted']
    assert outcome['member_trial_ids']==[t.trial_id for t in trials]
    assert outcome['weights']==[.5,.5]
    assert outcome['assessed_score']<outcome['baseline_score']
    assert outcome['calibrated_oof_score']<1e-12
    np.testing.assert_allclose(pd.read_csv(outcome['prediction_path'])['y'],[3,5])
    rejected=choose_postprocessing(spec,split,trials,tmp_path,remaining_seconds=5,overhead_seconds=5)
    assert not rejected['accepted'] and 'time' in rejected['reason']


def test_foreign_split_never_enters_blending(tmp_path):
    spec=SimpleNamespace(task_type='regression',metric='rmse',maximize=False)
    result=ExperimentResult('trial_0001','ok',str(tmp_path),'code','wrong split',metric=1.,
                             model_details={'training':{'split_fingerprint':'foreign'}})
    plan=choose_postprocessing(spec,{'mode':'evaluate_cv','split_fingerprint':'current'},[result],tmp_path,
                               remaining_seconds=10,overhead_seconds=1)
    assert not plan['accepted'] and plan['reason']=='no_compatible_oof'
