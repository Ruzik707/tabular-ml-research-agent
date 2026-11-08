from types import SimpleNamespace

import pytest

from automl_agent.agent import ResearchRun
from automl_agent.config import parse_config
from automl_agent.contracts import ExperimentResult


def test_default_finish_keeps_researching_when_time_for_experiments_remains():
    research=ResearchRun.__new__(ResearchRun)
    research.config=parse_config(['--offline'])
    research.task=SimpleNamespace(maximize=False)
    research.budget=SimpleNamespace(research_remaining=lambda:1200)
    research.journal=SimpleNamespace(append=lambda *a,**k:None)
    research.results=[ExperimentResult('trial_0001','ok','.','base','baseline',metric=1.,is_baseline=True),
                      ExperimentResult('trial_0002','ok','.','llm','measured',metric=2.,seconds=30.)]
    with pytest.raises(ValueError,match='Time-first'):
        research._finish('Первый опыт готов')
    research.results.extend(ExperimentResult(f'trial_{i:04d}','ok','.',str(i),'stagnated',metric=2.,seconds=30.) for i in range(3,10))
    with pytest.raises(ValueError,match='Time-first'):
        research._finish('Восемь успешных гипотез не улучшили результат')
    research.budget=SimpleNamespace(research_remaining=lambda:40)
    assert research._finish('Оставшегося времени недостаточно для следующего опыта')['finished']


def test_reserve_covers_three_candidates_even_from_same_family():
    research=ResearchRun.__new__(ResearchRun)
    research.config=parse_config(['--offline'])
    research.task=SimpleNamespace(maximize=False,validation_fraction=.2)
    research.profile={'train_shape':[1000,3]}
    research.split={'mode':'evaluate_cv'}
    research.budget=SimpleNamespace(reserve_seconds=0)
    research.results=[ExperimentResult(f'trial_{i:04d}','ok','.',str(i),'CV',metric=float(i),seconds=900,
                        recipe={'family':'CatBoost'},model_details={'training':{'estimated_refit_seconds':100}})
                      for i in range(1,4)]
    research._update_reserve(record=False)
    assert research.reserve_plan['potential_ensemble_fit_seconds']==450
    assert research.budget.reserve_seconds==518  # 450 fit + 8 startup + 60 bounded write overhead
