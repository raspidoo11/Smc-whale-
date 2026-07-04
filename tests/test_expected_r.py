"""Regression test for the expected-R feature_names mismatch seen in prod:
the expected-R model must use its OWN feature list, not the classifier's, so a
drifted classifier feature set can't break its prediction."""

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import xgboost_trainer as xt


def test_expected_r_uses_own_feature_list_not_classifier(tmp_path, monkeypatch):
    feats = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=feats)
    y = X["a"] * 2 - X["b"]
    model = XGBRegressor(n_estimators=10, max_depth=2, verbosity=0)
    model.fit(X, y)

    rmodel_path = tmp_path / "r.pkl"
    rfeat_path = tmp_path / "rfeat.pkl"
    clf_feat_path = tmp_path / "clf.pkl"
    joblib.dump(model, rmodel_path)
    joblib.dump(feats, rfeat_path)
    # Simulate the prod bug: the classifier's feature list has drifted and now
    # carries an extra old feature the expected-R model was NOT trained on.
    joblib.dump(feats + ["sl_tightness"], clf_feat_path)

    monkeypatch.setattr(xt, "EXPECTED_R_MODEL_PATH", str(rmodel_path))
    monkeypatch.setattr(xt, "EXPECTED_R_FEATURE_PATH", str(rfeat_path))
    monkeypatch.setattr(xt, "FEATURE_PATH", str(clf_feat_path))

    # Extra/absent keys must be reconciled by reindex; drift must NOT raise.
    val = xt.get_expected_r({"a": 1.0, "b": 0.5, "c": -0.2, "sl_tightness": 9.9, "extra": 1})
    assert val is not None
    assert isinstance(val, float)


def test_expected_r_none_without_its_feature_list(tmp_path, monkeypatch):
    monkeypatch.setattr(xt, "EXPECTED_R_MODEL_PATH", str(tmp_path / "missing.pkl"))
    monkeypatch.setattr(xt, "EXPECTED_R_FEATURE_PATH", str(tmp_path / "missing2.pkl"))
    assert xt.get_expected_r({"a": 1}) is None
