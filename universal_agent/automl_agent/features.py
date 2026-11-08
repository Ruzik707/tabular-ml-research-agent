"""Bounded declarative features; every learned statistic belongs to fit rows.

Operation vocabulary follows oleg/rakovskiy. Unlike their combined train+test
transform, this sklearn transformer never receives test data in fit or uses y.
This module is embedded into generated candidate source for portable reloads.
"""
import json
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


FEATURE_OPERATIONS = {
    'drop': {'cols': ['column']},
    'to_nan': {'cols': ['column'], 'values': [0, -1]},
    'log1p': {'cols': ['column']},
    'missing_flags': {'cols': ['column']},
    'freq_encode': {'cols': ['category']},
    'group_stats': {'by': ['category'], 'cols': ['number'], 'stats': ['mean', 'std']},
    'ratio': {'pairs': [['numerator', 'denominator']]},
    'diff': {'pairs': [['left', 'right']]},
    'product': {'pairs': [['left', 'right']]},
    'row_stats': {'cols': ['number1', 'number2'], 'stats': ['mean', 'std', 'min', 'max']},
    'combine': {'cols': ['category1', 'category2']},
    'bin': {'cols': ['number'], 'q': 10},
    'text_stats': {'cols': ['text'], 'top_k': 16},
    'date_parts': {'cols': ['date']},
}


def _feature_columns(value, label):
    if (not isinstance(value, list) or not 1 <= len(value) <= 32
            or any(not isinstance(c, str) or not c or len(c) > 160 for c in value)
            or len(set(value)) != len(value)):
        raise ValueError(label + ' must contain 1..32 unique column names')
    return list(value)


def validate_feature_ops(ops):
    if not isinstance(ops, list) or len(ops) > 16:
        raise ValueError('feature_ops must be an array of at most 16 operations')
    result = []
    for item in ops:
        if not isinstance(item, dict) or not isinstance(item.get('op'), str) or item['op'] not in FEATURE_OPERATIONS:
            raise ValueError('Unknown feature operation; arbitrary code is not supported')
        kind = item['op']
        allowed = {'op', *FEATURE_OPERATIONS[kind]}
        if set(item) - allowed:
            raise ValueError('Unknown feature operation field')
        op = {'op': kind}
        if kind in {'ratio', 'diff', 'product'}:
            pairs = item.get('pairs')
            if not isinstance(pairs, list) or not 1 <= len(pairs) <= 16:
                raise ValueError('pairs must contain 1..16 column pairs')
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Each pair must contain two columns')
                for column in pair:
                    _feature_columns([column], 'pair column')
            op['pairs'] = [list(pair) for pair in pairs]
        else:
            op['cols'] = _feature_columns(item.get('cols'), 'cols')
        if kind == 'group_stats':
            op['by'] = _feature_columns(item.get('by'), 'by')
        if kind in {'row_stats', 'group_stats'}:
            stats = item.get('stats', ['mean', 'std'])
            supported = {'mean', 'std', 'min', 'max', 'median'}
            if not isinstance(stats, list) or not stats or any(not isinstance(s, str) or s not in supported for s in stats) or len(set(stats)) != len(stats):
                raise ValueError('stats must be unique mean/std/min/max/median values')
            op['stats'] = list(stats)
        if kind == 'to_nan':
            values = item.get('values')
            if not isinstance(values, list) or not 1 <= len(values) <= 16:
                raise ValueError('values must contain 1..16 scalar values')
            if any(type(v) not in (str, int, float, bool) or (isinstance(v, (float, int)) and not math.isfinite(v)) for v in values):
                raise ValueError('values must be finite JSON scalars')
            op['values'] = list(values)
        if kind == 'bin':
            q = item.get('q', 10)
            if type(q) is not int or not 2 <= q <= 32:
                raise ValueError('q must be an integer in [2,32]')
            op['q'] = q
        if kind == 'text_stats':
            top_k = item.get('top_k', 16)
            if type(top_k) is not int or not 0 <= top_k <= 32:
                raise ValueError('top_k must be an integer in [0,32]')
            op['top_k'] = top_k
        result.append(op)
    return result


def _feature_outputs(op, index):
    prefix = '__ua_' + str(index) + '_'
    kind = op['op']
    if kind in {'drop', 'to_nan'}:
        return []
    if kind in {'ratio', 'diff', 'product'}:
        return [prefix + str(i) + '_' + kind for i in range(len(op['pairs']))]
    if kind == 'combine':
        return [prefix + 'combined']
    if kind == 'row_stats':
        return [prefix + 'row_' + stat for stat in op['stats']]
    if kind == 'group_stats':
        return [prefix + col + '_' + stat for col in op['cols'] for stat in op['stats']]
    if kind == 'text_stats':
        return [prefix + c + '_' + suffix for c in op['cols']
                for suffix in ['chars', 'words'] + ['token_'+str(j) for j in range(op['top_k'])]]
    if kind == 'date_parts':
        return [prefix + col + '_' + suffix for col in op['cols']
                for suffix in ['epoch_days', 'month', 'weekday']]
    suffix = {'freq_encode': 'freq', 'missing_flags': 'missing', 'log1p': 'log1p', 'bin': 'bin'}[kind]
    return [prefix + col + '_' + suffix for col in op['cols']]


def feature_schema(columns, categorical, ops):
    columns, cats = list(columns), list(categorical)
    generated = 0
    for i, op in enumerate(validate_feature_ops(ops)):
        referenced = op.get('cols', []) + op.get('by', []) + [c for pair in op.get('pairs', []) for c in pair]
        missing = [c for c in referenced if c not in columns]
        if missing:
            raise ValueError('Unknown feature columns: ' + str(missing))
        added = _feature_outputs(op, i)
        if any(c in columns for c in added):
            raise ValueError('Generated feature name collides with an input column')
        if op['op'] == 'drop':
            columns = [c for c in columns if c not in op['cols']]
            cats = [c for c in cats if c in columns]
        columns.extend(added)
        if op['op'] == 'combine':
            cats.extend(added)
        generated += len(added)
        if generated > 128:
            raise ValueError('At most 128 generated features are allowed')
    if not columns:
        raise ValueError('Feature operations removed every model feature')
    return columns, cats


def _numeric_feature(frame, column):
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors='coerce')
    if (raw.notna() & numeric.isna()).any():
        raise ValueError('Feature operation requires numeric values in column '+str(column))
    return numeric.replace([np.inf, -np.inf], np.nan)


def _parse_iso_dates(values):
    # Explicit ISO8601 parsing avoids pandas' batch-dependent format inference.
    # UTC normalization makes timezone-aware and naive inputs comparable. A
    # calendar day is required: ISO partial dates such as "2024" and "2024-01"
    # are rejected rather than silently assigned an invented day. Basic
    # YYYYMMDD and extended YYYY-MM-DD forms, optionally followed by a time,
    # are accepted; pandas still validates the actual calendar/time values.
    # Parse text rather than raw numeric scalars: pandas otherwise interprets
    # integers as epoch units (for example, 20240101 becomes nanoseconds after
    # the epoch). Native datetime64 columns are first rendered as text too, so
    # their storage unit cannot change the parsed epoch-unit arithmetic.
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        text = values.astype('string')
    else:
        text = values.map(lambda value: value if isinstance(value, str) else pd.NA).astype('string')
    parsed = pd.to_datetime(text, format='ISO8601', utc=True, errors='coerce')
    has_day = text.str.match(r'^(?:\d{4}-\d{2}-\d{2}|\d{8})(?:[Tt ].*)?$', na=False)
    return parsed.where(has_day)


def _feature_keys(frame, columns):
    # JSON tuples avoid collisions such as ('a|b','c') vs ('a','b|c').
    arrays = [frame[c].map(lambda v: [0] if pd.isna(v) else [1, str(v)]).tolist() for c in columns]
    return pd.Series([json.dumps(row, ensure_ascii=False, separators=(',', ':'))
                      for row in zip(*arrays)], index=frame.index)


class FeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, ops=None, categorical_columns=None):
        self.ops = ops
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.operations_ = validate_feature_ops(self.ops or [])
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.output_columns_, self.output_categories_ = feature_schema(
            list(X.columns), self.categorical_columns or [], self.operations_)
        self.statistics_ = {}
        self._apply(X, fitting=True)
        return self

    def transform(self, X):
        if not hasattr(self, 'statistics_'):
            raise ValueError('FeatureTransformer has not been fitted')
        return self._apply(X, fitting=False)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_columns_, dtype=object)

    def _apply(self, X, fitting):
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Feature input schema changed since fit')
        frame = X.copy()
        for i, op in enumerate(self.operations_):
            kind, added = op['op'], _feature_outputs(op, i)
            if kind == 'drop':
                frame = frame.drop(columns=op['cols'])
            elif kind == 'to_nan':
                for c in op['cols']:
                    frame[c] = frame[c].mask(frame[c].isin(op['values']))
            elif kind == 'combine':
                frame[added[0]] = _feature_keys(frame, op['cols'])
            elif kind == 'row_stats':
                values = pd.DataFrame({c: _numeric_feature(frame, c) for c in op['cols']})
                for name, stat in zip(added, op['stats']):
                    frame[name] = values.agg(stat, axis=1)
            elif kind in {'ratio', 'diff', 'product'}:
                for name, (a, b) in zip(added, op['pairs']):
                    left, right = _numeric_feature(frame, a), _numeric_feature(frame, b)
                    if kind == 'ratio':
                        value = left / right.mask(right == 0)
                    elif kind == 'diff':
                        value = left - right
                    else:
                        value = left * right
                    frame[name] = value.replace([np.inf, -np.inf], np.nan)
            elif kind == 'group_stats':
                keys = _feature_keys(frame, op['by'])
                for name, (c, stat) in zip(added, [(c, s) for c in op['cols'] for s in op['stats']]):
                    if fitting:
                        values = _numeric_feature(frame, c)
                        self.statistics_[name] = values.groupby(keys, sort=False).agg(stat).to_dict()
                    frame[name] = keys.map(self.statistics_[name])
            elif kind == 'date_parts':
                for c in op['cols']:
                    raw = frame[c]
                    if fitting and pd.api.types.is_numeric_dtype(raw.dtype):
                        raise ValueError('date_parts requires a non-numeric ISO8601 date column: ' + str(c))
                    parsed = _parse_iso_dates(raw)
                    if fitting:
                        nonnull = int(raw.notna().sum())
                        valid = int((raw.notna() & parsed.notna()).sum())
                        if nonnull == 0:
                            raise ValueError('date_parts requires at least one non-null fit date in column ' + str(c))
                        if valid / nonnull <= 0.95:
                            raise ValueError('date_parts requires more than 95% parseable non-null ISO8601 fit dates in column ' + str(c))
                    prefix = '__ua_' + str(i) + '_' + c + '_'
                    # Use the parsed array's own integer unit. Flooring the
                    # Timestamp itself can overflow at pandas' minimum valid
                    # nanosecond timestamp; integer floor division is safe and
                    # also handles dates before the Unix epoch correctly.
                    units_per_day = {
                        's': 86_400,
                        'ms': 86_400_000,
                        'us': 86_400_000_000,
                        'ns': 86_400_000_000_000,
                    }[parsed.dtype.unit]
                    epoch_days = parsed.astype('int64') // units_per_day
                    epoch_days = epoch_days.astype(float).mask(parsed.isna())
                    frame[prefix + 'epoch_days'] = epoch_days
                    frame[prefix + 'month'] = parsed.dt.month.astype(float)
                    frame[prefix + 'weekday'] = parsed.dt.weekday.astype(float)
            elif kind == 'text_stats':
                for c in op['cols']:
                    prefix = '__ua_' + str(i) + '_' + c + '_'
                    text = frame[c].fillna('').astype(str).str.lower().str.slice(0, 10000)
                    tokens = text.str.findall(r'\w+')
                    if fitting:
                        counts = {}
                        # Bound vocabulary construction as well as output width.
                        for row in tokens.iloc[:10000]:
                            for token in set(row):
                                counts[token] = counts.get(token, 0) + 1
                        self.statistics_[prefix] = sorted(counts, key=lambda t: (-counts[t], t))[:op['top_k']]
                    vocab = self.statistics_[prefix]
                    frame[prefix+'chars'] = text.str.len().astype(float)
                    frame[prefix+'words'] = tokens.str.len().astype(float)
                    for j in range(op['top_k']):
                        token = vocab[j] if j < len(vocab) else None
                        frame[prefix+'token_'+str(j)] = tokens.map(lambda row: row.count(token) if token is not None else 0).astype(float)
            else:
                for name, c in zip(added, op['cols']):
                    if kind == 'missing_flags':
                        frame[name] = frame[c].isna().astype(float)
                    elif kind == 'log1p':
                        values = _numeric_feature(frame, c)
                        frame[name] = np.sign(values) * np.log1p(np.abs(values))
                    elif kind == 'freq_encode':
                        keys = _feature_keys(frame, [c])
                        if fitting:
                            self.statistics_[name] = keys.value_counts(normalize=True).to_dict()
                        frame[name] = keys.map(self.statistics_[name]).fillna(0.)
                    elif kind == 'bin':
                        values = _numeric_feature(frame, c)
                        if fitting:
                            valid = values.dropna()
                            edges = np.unique(valid.quantile(np.linspace(0, 1, op['q']+1)).to_numpy()) if len(valid) else np.array([])
                            self.statistics_[name] = edges[1:-1]
                        frame[name] = np.where(values.isna(), np.nan, np.searchsorted(self.statistics_[name], values, side='right'))
        return frame.loc[:, self.output_columns_]
