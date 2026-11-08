import json

from automl_agent.audit import AuditJournal


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_projects_bounded_dataset_drift_and_temporal_facts(tmp_path):
    journal = AuditJournal(tmp_path / "events.jsonl", echo=False)
    journal.append("dataset_diagnosed", diagnostics={
        "adversarial_cv": {
            "status": "ok", "roc_auc_mean": 0.91, "roc_auc_std": 0.02, "folds": 3, "seconds": 6.4,
            "without_temporal": {"status": "ok", "roc_auc_mean": 0.57},
            "messages": ["private payload"] * 200,
        },
        "temporal_candidates": [{
            "feature": "Year_Factor", "kind": "numeric", "future_only": True,
            "direction": "future", "train_group_count": 5,
            "train_min": 1, "train_max": 5, "test_min": 6, "test_max": 6,
            "raw_values": ["private row"] * 100,
        }],
        "features": {
            f"column_{i}": {
                "kind": "categorical", "train_missing_rate": 0.0, "test_missing_rate": 0.2,
                "category_shift": {
                    "train_only_count": 1, "test_only_count": 2,
                    "test_only_top": [{"value": "private category", "count": 17}],
                },
            }
            for i in range(40)
        },
        "chain_of_thought": "private hidden reasoning",
    })

    projected = _records(tmp_path / "decisions.jsonl")[0]["details"]
    assert projected["adversarial_cv"]["roc_auc_mean"] == 0.91
    assert projected["adversarial_cv"]["without_temporal"]["roc_auc_mean"] == 0.57
    assert projected["temporal_candidates"][0]["feature"] == "Year_Factor"
    assert projected["temporal_candidates"][0]["test_min"] == 6
    assert len(projected["feature_shift"]) <= 8
    rendered = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "0.91" in rendered and "0.57" in rendered
    assert "Year_Factor" in rendered and "train max=5" in rendered and "test min=6" in rendered
    assert "6.4 с" in rendered
    assert "private category" not in rendered
    assert "private hidden reasoning" not in rendered
    assert len(rendered) < 1800


def test_validation_plan_shows_support_policy_folds_coverage_and_warnings(tmp_path):
    journal = AuditJournal(tmp_path / "events.jsonl", echo=False)
    journal.append("validation_planned", summary={
        "seed": 42,
        "support_policy": "forward_group_cv",
        "fold_count": 4,
        "coverage": {"eligible_rows": 100, "oof_rows": 80, "fraction": 0.8},
        "warnings": ["first warning", "second warning"] + ["extra"] * 30,
        "private_path": "/private/path",
    })

    record = _records(tmp_path / "decisions.jsonl")[0]
    details = record["details"]["summary"]
    assert details["support_policy"] == "forward_group_cv"
    assert details["fold_count"] == 4
    assert details["coverage"]["oof_rows"] == 80
    assert len(details["warnings"]) <= 5
    human = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "forward_group_cv" in human and "4" in human and "80" in human
    assert "/private/path" not in human
    assert len(human) < 900


def test_postprocessing_projection_explains_ensemble_choice_compactly(tmp_path):
    journal = AuditJournal(tmp_path / "events.jsonl", echo=False)
    journal.append("postprocessing_selected", plan={"accepted": True, "private_note": "hidden"},
                   member_trial_ids=["trial_1", "trial_2", "trial_3", "trial_4"],
                   weights=[0.65, 0.35, 0, 0],
                   class_multipliers={"rare": 1.2, "common": 0.9},
                   baseline_score=0.61, assessed_score=0.67, calibrated_oof_score=0.65,
                   assessment="accepted: assessed score improved on the selection split",
                   warnings=["no material warning"] * 20,
                   hidden_reasoning="do not expose")

    details = _records(tmp_path / "decisions.jsonl")[0]["details"]
    assert details["accepted"] is True
    assert details["member_trial_ids"] == ["trial_1", "trial_2", "trial_3"]
    assert details["weights"] == [0.65, 0.35, 0]
    assert details["baseline_score"] == 0.61
    assert details["assessed_score"] == 0.67
    assert details["calibrated_oof_score"] == 0.65
    assert len(details["warnings"]) <= 4
    human = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "принята" in human and "trial_1" in human and "0.67" in human
    assert "assessed score improved" in human
    assert "hidden" not in human and "do not expose" not in human
    assert len(human) < 1200


def test_trial_completion_projects_bounded_diagnostic_summary_only(tmp_path, capsys):
    journal = AuditJournal(tmp_path / "events.jsonl", echo=True)
    journal.append("trial_completed", result={
        "trial_id": "trial_9", "status": "ok", "metric": 0.67, "seconds": 12.5,
        "diagnostics": {
            "per_class": [
                {"label": f"class_{i}", "f1": i / 50, "recall": 0.1, "precision": 0.2, "support": 5,
                 "private_prompt": "must not appear"}
                for i in range(50)
            ],
            "total_classes": 50, "partial": True,
            "confusion_matrix": {
                "labels": list(range(30)),
                "matrix": [[i + j for j in range(30)] for i in range(30)],
                "total_classes": 50, "partial": True,
            },
            "segments": [
                {"feature": "region", "category": f"region_{i}", "count": 25,
                 "rmse": 20 - i, "mae": 15 - i}
                for i in range(40)
            ],
            "total_segments": 40, "partial": True,
            "validation": {"metric_name": "macro_f1", "score": 0.67, "seconds": 4.2,
                           "response": "private response body"},
            "feature_importance": {
                "status": "ok", "method": "native_feature_importances_",
                "items": [{"feature": f"x{i}", "importance": 1 / (i + 1)} for i in range(50)],
                "interpretation": "association only",
            },
            "raw_predictions": [[1, 2]] * 1000,
            "chain_of_thought": "hidden reasoning",
        },
    })

    record = _records(tmp_path / "decisions.jsonl")[0]
    projected = record["details"]["diagnostics"]
    assert len(projected["per_class"]) <= 8
    assert len(projected["confusion_matrix"]["labels"]) <= 10
    assert len(projected["confusion_matrix"]["matrix"]) <= 10
    assert len(projected["segments"]) <= 5
    assert len(projected["feature_importance"]["items"]) <= 8
    assert projected["validation"]["score"] == 0.67
    content = (tmp_path / "agent.log").read_text(encoding="utf-8") + json.dumps(record, ensure_ascii=False)
    assert "private prompt" not in content and "private response body" not in content
    assert "hidden reasoning" not in content and "raw_predictions" not in content
    human = (tmp_path / "agent.log").read_text(encoding="utf-8")
    assert "macro_f1" in human and "region_0" in human and "x0" in human
    assert len(human) < 2000
    terminal = capsys.readouterr().err
    assert terminal == human
    assert len(terminal) < 2000


def test_actual_nested_postprocessing_plan_projects_scores_and_multiplier_axis(tmp_path):
    journal=AuditJournal(tmp_path/'events.jsonl',echo=False)
    journal.append('postprocessing_selected',plan={'accepted':True,'member_trial_ids':['trial_1','trial_2'],
        'weights':[.75,.25],'classes':['common','rare'],'class_multipliers':[.75,1.5],
        'baseline_score':.6,'assessed_score':.67,'calibrated_oof_score':.7,'assessment':'cross_fitted',
        'reason':'meta_improvement','warnings':['not nested CV']})
    projected=_records(tmp_path/'decisions.jsonl')[0]['details']
    assert projected['member_trial_ids']==['trial_1','trial_2']
    assert projected['weights']==[.75,.25]
    assert projected['assessed_score']==.67
    assert projected['class_multipliers']==[{'label':'common','multiplier':.75},{'label':'rare','multiplier':1.5}]
    assert projected['reason']=='meta_improvement'
    text=(tmp_path/'agent.log').read_text()
    assert '0.67' in text and 'rare' in text


def test_binary_threshold_and_positive_class_are_visible_in_compact_audit(tmp_path):
    journal = AuditJournal(tmp_path / 'events.jsonl', echo=False)
    journal.append('postprocessing_selected', plan={
        'accepted': True, 'member_trial_ids': ['trial_1'], 'weights': [1.],
        'binary_threshold': 0., 'positive_class': 'alert',
        'baseline_score': .2, 'assessed_score': .5, 'calibrated_oof_score': .6,
    })
    projected = _records(tmp_path / 'decisions.jsonl')[0]['details']
    assert projected['binary_threshold'] == 0.
    assert projected['positive_class'] == 'alert'
    text = (tmp_path / 'agent.log').read_text()
    assert 'порог' in text and 'alert' in text


def test_family_comparison_log_does_not_dump_fold_training_payload(tmp_path):
    journal=AuditJournal(tmp_path/'events.jsonl',echo=False)
    journal.append('family_comparison_completed',comparison={'selected_trial':'trial_1','metric':'rmse',
        'trials':[{'family':'CatBoost','trial_id':'trial_1','status':'ok','metric':3.5,'seconds':2,
            'model_details':{'class':'CatBoostRegressor','parameters':{'iterations':777},
                'training':{'folds':[{'timing':123,'details':['value']*1000}]*5}}}]})
    text=(tmp_path/'agent.log').read_text()
    assert 'CatBoost' in text and '3.5' in text and '777' in text
    assert len(text)<1500
