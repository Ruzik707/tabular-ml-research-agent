"""Behavioral tests for the explicit, fit-validated date feature operation."""

import sys
import types

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.features import FeatureTransformer, feature_schema, validate_feature_ops


def test_date_parts_generates_utc_epoch_month_weekday_and_preserves_input():
    ops = [{"op": "date_parts", "cols": ["when"]}]
    frame = pd.DataFrame(
        {
            "when": ["2024-01-01T01:00:00-02:00", "2024-01-02T01:00:00+03:00"],
            "amount": [2.0, 3.0],
        }
    )

    transformer = FeatureTransformer(ops).fit(frame)
    result = transformer.transform(frame)

    assert list(result.columns) == [
        "when", "amount", "__ua_0_when_epoch_days", "__ua_0_when_month", "__ua_0_when_weekday"
    ]
    # Both instants are 2024-01-01 in UTC despite having different local dates.
    assert result["__ua_0_when_epoch_days"].tolist() == [19723.0, 19723.0]
    assert result["__ua_0_when_month"].tolist() == [1.0, 1.0]
    assert result["__ua_0_when_weekday"].tolist() == [0.0, 0.0]
    pd.testing.assert_series_equal(result["when"], frame["when"])


def test_date_parts_transform_maps_missing_ambiguous_and_out_of_range_to_nan_without_refit():
    ops = [{"op": "date_parts", "cols": ["when"]}]
    train = pd.DataFrame({"when": ["2020-02-29", "2021-03-01"]})
    transformer = FeatureTransformer(ops).fit(train)
    learned_state = dict(transformer.statistics_)
    probe = pd.DataFrame(
        {
            "when": [None, "not-a-date", "01/02/2024", "2500-01-01",
                     "2024-01-01T00:00:00+25:00", "2020-02-29"]
        }
    )

    result = transformer.transform(probe)

    derived = result.filter(like="__ua_")
    assert derived.iloc[:5].isna().all().all()
    assert derived.iloc[5].tolist() == [18321.0, 2.0, 5.0]
    assert transformer.statistics_ == learned_state
    # Results are rowwise and do not depend on what other test rows share the batch.
    one_row = transformer.transform(probe.iloc[[5]])
    pd.testing.assert_frame_equal(derived.iloc[[5]].reset_index(drop=True),
                                  one_row.filter(like="__ua_").reset_index(drop=True))


def test_date_parts_never_parses_numeric_transform_values_as_epoch_or_compact_text():
    transformer = FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit(
        pd.DataFrame({"when": ["2024-01-01", "2024-01-02"]})
    )
    probe = pd.DataFrame({"when": pd.Series([20240101, "20240101", "2024-01-01"], dtype=object)})

    result = transformer.transform(probe)

    assert result.loc[0, "__ua_0_when_epoch_days"] != result.loc[0, "__ua_0_when_epoch_days"]
    assert result["__ua_0_when_epoch_days"].iloc[1:].tolist() == [19723.0, 19723.0]


def test_date_parts_normalizes_datetime64_units_and_maps_nat_out_of_range_to_nan():
    transformer = FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit(
        pd.DataFrame({"when": ["2024-01-01", "2024-01-02"]})
    )
    probe = pd.DataFrame(
        {
            "when": np.array(
                ["2024-01-01", "1969-12-31", "2500-01-01", "NaT"], dtype="datetime64[s]"
            )
        }
    )

    result = transformer.transform(probe)

    assert result["__ua_0_when_epoch_days"].iloc[:2].tolist() == [19723.0, -1.0]
    assert result.filter(like="__ua_").iloc[2:].isna().all().all()


@pytest.mark.parametrize(
    "values, message",
    [
        ([None, None], "at least one non-null"),
        ([20200101, 20200102], "numeric"),
        (["2024-01-01"] * 19 + ["01/02/2024"], "95%"),
        (["2024"] * 20, "95%"),
    ],
)
def test_date_parts_rejects_unusable_fit_column(values, message):
    frame = pd.DataFrame({"when": values})
    with pytest.raises(ValueError, match=message):
        FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit(frame)


def test_date_parts_schema_is_bounded_and_operation_is_catalogued():
    op = {"op": "date_parts", "cols": ["when"]}
    assert validate_feature_ops([op]) == [op]
    assert feature_schema(["when"], [], [op])[0] == [
        "when", "__ua_0_when_epoch_days", "__ua_0_when_month", "__ua_0_when_weekday"
    ]

    columns = [f"date_{i}" for i in range(43)]
    with pytest.raises(ValueError, match="128 generated features"):
        feature_schema(
            columns,
            [],
            [{"op": "date_parts", "cols": columns[:32]},
             {"op": "date_parts", "cols": columns[32:]}],
        )


def test_date_parts_handles_timestamp_integer_boundaries_and_pre_epoch_days():
    frame = pd.DataFrame(
        {
            "when": [
                "1677-09-21T00:12:43.145224193Z",  # pandas Timestamp.min
                "1969-12-31T23:59:59.999999999Z",  # one nanosecond before epoch
                "2262-04-11T23:47:16.854775807Z",  # pandas Timestamp.max
            ]
        }
    )

    result = FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit_transform(frame)

    assert result["__ua_0_when_epoch_days"].tolist() == [-106752.0, -1.0, 106751.0]
    assert result["__ua_0_when_month"].tolist() == [9.0, 12.0, 4.0]


def test_date_parts_accepts_compact_iso_calendar_day_but_rejects_partial_date():
    compact = pd.DataFrame({"when": ["20240101", "20240229"]})
    result = FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit_transform(compact)
    assert result["__ua_0_when_epoch_days"].tolist() == [19723.0, 19782.0]

    partial = pd.DataFrame({"when": ["2024-01"] * 20})
    with pytest.raises(ValueError, match="95%"):
        FeatureTransformer([{"op": "date_parts", "cols": ["when"]}]).fit(partial)


def test_generated_recipe_with_date_parts_is_standalone_and_joblib_reloadable(tmp_path):
    from automl_agent.recipes import default_recipe, recipe_code

    recipe = default_recipe("Linear", "regression")
    recipe["preprocessing"]["feature_ops"] = [{"op": "date_parts", "cols": ["when"]}]
    source = recipe_code(recipe, "regression", n_jobs=1)
    assert "from automl_agent" not in source
    module_name = "generated_date_parts_candidate"
    module = types.ModuleType(module_name)
    exec(compile(source, "<generated-date-parts>", "exec"), module.__dict__)
    sys.modules[module_name] = module
    cloudpickle.register_pickle_by_value(module)
    try:
        context = {
            "task_type": "regression",
            "feature_columns": ["when", "x"],
            "categorical_columns": ["when"],
            "n_jobs": 1,
        }
        model = module.build_model(context)
        X = pd.DataFrame(
            {"when": pd.date_range("2022-01-01", periods=24, freq="D").astype(str),
             "x": np.arange(24, dtype=float)}
        )
        y = np.arange(24, dtype=float) * 0.5
        model.fit(X, y)
        probe = pd.DataFrame({"when": ["2024-02-29T23:30:00-02:00", None], "x": [5.0, 8.0]})
        prediction = model.predict(probe)
        artifact = tmp_path / "date-model.joblib"
        joblib.dump(model, artifact)
        restored = joblib.load(artifact)
        np.testing.assert_allclose(restored.predict(probe), prediction)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
        sys.modules.pop(module_name, None)
