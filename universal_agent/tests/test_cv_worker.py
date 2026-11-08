import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import TaskSpec
from automl_agent.recipes import recipe_code
from automl_agent.validation import prepare_cv_split
from automl_agent.worker import _run_manifest


def fixture(tmp_path, classification=False):
    train = pd.DataFrame({'x': np.arange(100)/10, 'cat': ['a','b']*50,
                          'y': ['negative','positive']*50 if classification else np.arange(100)*2.})
    train.to_csv(tmp_path/'train.csv', index=False)
    train.drop(columns='y').iloc[:3].to_csv(tmp_path/'test.csv', index=False)
    task = TaskSpec('task.md', str(tmp_path/'train.csv'), str(tmp_path/'test.csv'), 'y',
                    'classification' if classification else 'regression', 'macro_f1' if classification else 'rmse',
                    classification, ['x','cat'], [], ['cat'], ['y'], 'hash', classes=['negative','positive'] if classification else [])
    split = prepare_cv_split(task, tmp_path/'cache', folds=5, strategy='iid')
    code = tmp_path/'code.py'
    code.write_text(recipe_code({'family':'ExtraTrees','parameters':{'n_estimators':10}}, task.task_type, n_jobs=1))
    manifest = {'mode':'evaluate_cv','context':task.model_context(1),
                'inputs':{k:split[k] for k in ['train_features','train_target','test_features']},
                'cv_plan':split['cv_plan'],'artifact_dir':str(tmp_path/'artifacts'),
                'code_path':str(code),'save_model':True,'preflight':True}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return task, split, path, manifest


@pytest.mark.parametrize('classification',[False,True])
def test_worker_oof_and_saved_fold_ensemble(tmp_path, classification):
    task, split, path, manifest = fixture(tmp_path,classification)
    result = _run_manifest(path)
    directory=Path(manifest['artifact_dir'])
    assert result['validation_rows']==100
    assert len(result['estimator_details']['training']['folds'])==5
    assert result['preflight']['status']=='ok'
    assert len(pd.read_csv(directory/'valid_predictions.csv'))==100
    model=joblib.load(directory/'model.joblib')
    X=pd.read_csv(task.test_path)
    np.testing.assert_array_equal(model.predict(X),pd.read_csv(directory/'test_predictions.csv')['y'].to_numpy())
    if classification:
        probabilities=np.load(directory/'valid_probabilities.npy')
        assert probabilities.shape==(100,2)
        np.testing.assert_allclose(probabilities.sum(axis=1),1)
        assert json.loads((directory/'probability_classes.json').read_text())['classes']==task.classes


def test_preflight_catches_shape_bug_before_cv(tmp_path):
    _,_,path,manifest=fixture(tmp_path)
    Path(manifest['code_path']).write_text('''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

def build_model(context):
    return make_pipeline(FunctionTransformer(lambda X: np.asarray(X)["x"]), ExtraTreesRegressor(n_estimators=10))
''')
    with pytest.raises(Exception):
        _run_manifest(path)
    directory=Path(manifest['artifact_dir'])
    preflight=json.loads((directory/'preflight.json').read_text())
    assert preflight['status']=='error'
    assert not (directory/'fold_0.json').exists()


def test_host_rejects_reordered_oof_artifacts(tmp_path):
    from automl_agent.evaluation import score_trial
    task,split,path,manifest=fixture(tmp_path)
    _run_manifest(path)
    directory=Path(manifest['artifact_dir'])
    indices=np.load(directory/'oof_indices.npy')
    np.save(directory/'oof_indices.npy',indices[::-1])
    with pytest.raises(ValueError,match='OOF|indices|alignment'):
        score_trial(task,split,directory)


def test_alignment_rejects_out_of_range_probabilities():
    from automl_agent.prediction import aligned_probabilities
    class Invalid:
        classes_=['a','b']
        def predict_proba(self,X):return np.array([[2.,1.]])
    with pytest.raises(ValueError,match='probability'):
        aligned_probabilities(Invalid(),pd.DataFrame({'x':[1]}),['a','b'])


def test_free_code_custom_fit_predict_survives_cv_preflight(tmp_path):
    _,_,path,manifest=fixture(tmp_path)
    Path(manifest['code_path']).write_text('''import numpy as np
class SimpleModel:
    def fit(self,X,y):
        self.mean=float(np.mean(y))
        return self
    def predict(self,X): return np.full(len(X),self.mean)
def build_model(context):return SimpleModel()
''')
    result=_run_manifest(path)
    assert result['status']=='ok' and result['preflight']['status']=='ok'


def test_cv_fit_summary_preserves_native_thread_metadata(tmp_path):
    _,_,path,manifest=fixture(tmp_path)
    Path(manifest['code_path']).write_text(recipe_code({'family':'CatBoost','parameters':{'iterations':10}},'regression',n_jobs=1))
    result=_run_manifest(path)
    assert result['estimator_details']['parameters']['thread_count']==1
