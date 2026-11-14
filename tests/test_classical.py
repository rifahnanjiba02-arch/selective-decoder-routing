"""Offline tests for router.models.classical (Task 3). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from router.models import classical as C

ALLOWED = ["detection_event_count", "event_density", "activity_0", "activity_1",
           "distance", "rounds", "basis"]


def make_xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    cnt = rng.integers(0, 20, n)
    X = pd.DataFrame({
        "detection_event_count": cnt,
        "event_density": cnt / 40.0,
        "activity_0": rng.integers(0, 5, n),
        "activity_1": rng.integers(0, 5, n),
        "distance": rng.choice([3, 5], n),
        "rounds": rng.choice([3, 5], n),
        "basis": rng.choice(["X", "Z"], n),
    })
    y = ((cnt > 12) & (rng.random(n) < 0.7)).astype(int)
    y[:3] = [1, 0, 1]  # guarantee both classes
    return X, y


def avail(name):
    if name == "xgboost":
        pytest.importorskip("xgboost")
    return name


@pytest.fixture(scope="module")
def cfg():
    return C.load_model_config()


@pytest.mark.parametrize("name", C.MODEL_NAMES)
def test_scores_shape_range_alignment(name, cfg):
    avail(name)
    Xtr, ytr = make_xy(300, 0)
    Xva, _ = make_xy(80, 1)
    m = C.fit_router(name, Xtr, ytr, ALLOWED, cfg)
    s = C.predict_rescue_proba(m, Xva)
    assert s.shape == (len(Xva),)
    assert np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
    # row order preserved: scoring a single row equals the same row in the batch
    s1 = C.predict_rescue_proba(m, Xva.iloc[[5]])
    assert s1[0] == pytest.approx(s[5], abs=1e-9)


def test_all_four_models_same_rows(cfg):
    pytest.importorskip("xgboost")
    Xtr, ytr = make_xy(300, 0)
    Xva, _ = make_xy(60, 1)
    models = C.fit_all_routers(Xtr, ytr, ALLOWED, cfg)
    assert list(models) == list(C.MODEL_NAMES)
    scores = C.predict_all_routers(models, Xva)
    assert {len(v) for v in scores.values()} == {len(Xva)}


def test_dummy_is_train_prior(cfg):
    Xtr, ytr = make_xy()
    m = C.fit_router("dummy", Xtr, ytr, ALLOWED, cfg)
    s = C.predict_rescue_proba(m, make_xy(20, 2)[0])
    assert np.allclose(s, ytr.mean())


@pytest.mark.parametrize("bad", sorted(C.FORBIDDEN_COLUMNS))
def test_forbidden_column_rejected_before_fit(bad, cfg):
    Xtr, ytr = make_xy()
    Xtr[bad] = ytr
    with pytest.raises(ValueError, match="forbidden"):
        C.fit_router("logistic_regression", Xtr, ytr, ALLOWED, cfg)


def test_forbidden_column_rejected_at_predict(cfg):
    Xtr, ytr = make_xy()
    m = C.fit_router("logistic_regression", Xtr, ytr, ALLOWED, cfg)
    Xva, _ = make_xy(20, 3)
    Xva["label"] = 0
    with pytest.raises(ValueError, match="forbidden"):
        C.predict_rescue_proba(m, Xva)


def test_forbidden_in_allowlist_rejected(cfg):
    Xtr, ytr = make_xy()
    with pytest.raises(ValueError, match="forbidden"):
        C.fit_router("dummy", Xtr, ytr, ALLOWED + ["experiment"], cfg)


def test_unexpected_nonforbidden_column_never_reaches_model(cfg):
    Xtr, ytr = make_xy()
    m1 = C.fit_router("logistic_regression", Xtr, ytr, ALLOWED, cfg)
    Xtr2 = Xtr.assign(junk=ytr * 1000.0)
    m2 = C.fit_router("logistic_regression", Xtr2, ytr, ALLOWED, cfg)
    Xva, _ = make_xy(30, 4)
    assert np.allclose(C.predict_rescue_proba(m1, Xva),
                       C.predict_rescue_proba(m2, Xva.assign(junk=-5.0)))
    assert m2.named_steps["prep"].remainder == "drop"


def test_missing_allowed_column_rejected(cfg):
    Xtr, ytr = make_xy()
    with pytest.raises(ValueError, match="missing"):
        C.fit_router("dummy", Xtr.drop(columns=["distance"]), ytr, ALLOWED, cfg)


def test_preprocessing_state_comes_from_train_only(cfg):
    Xtr, ytr = make_xy()
    Xva, _ = make_xy(40, 5)
    Xva.loc[Xva.index[0], "basis"] = "Y"                    # category only in validation
    Xva.loc[Xva.index[1], "detection_event_count"] = 1e9    # extreme only in validation
    m = C.fit_router("logistic_regression", Xtr, ytr, ALLOWED, cfg)
    prep = m.named_steps["prep"]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    num_cols = [c for c in ALLOWED if c != "basis"]
    j = num_cols.index("detection_event_count")
    assert scaler.mean_[j] == pytest.approx(Xtr["detection_event_count"].mean())
    cats = prep.named_transformers_["cat"].named_steps["onehot"].categories_[0]
    assert sorted(cats) == ["X", "Z"]
    before = scaler.mean_.copy()
    s = C.predict_rescue_proba(m, Xva)          # unseen category must not crash
    assert np.isfinite(s).all()
    assert np.array_equal(before, scaler.mean_)  # predict did not refit anything


def test_basis_is_one_hot_encoded(cfg):
    Xtr, ytr = make_xy()
    m = C.fit_router("random_forest", Xtr, ytr, ALLOWED, cfg)
    out = m.named_steps["prep"].transform(Xtr.loc[:, ALLOWED])
    assert out.shape[1] == (len(ALLOWED) - 1) + 2


def test_target_validation(cfg):
    Xtr, ytr = make_xy()
    with pytest.raises(ValueError, match="rows"):
        C.fit_router("dummy", Xtr, ytr[:-1], ALLOWED, cfg)
    with pytest.raises(ValueError, match="0/1"):
        C.fit_router("dummy", Xtr, ytr + 1, ALLOWED, cfg)
    with pytest.raises(ValueError, match="both classes"):
        C.fit_router("dummy", Xtr, np.zeros(len(Xtr), int), ALLOWED, cfg)


def test_deterministic_given_seed(cfg):
    pytest.importorskip("xgboost")
    Xtr, ytr = make_xy()
    Xva, _ = make_xy(50, 6)
    for name in C.MODEL_NAMES:
        a = C.predict_rescue_proba(C.fit_router(name, Xtr, ytr, ALLOWED, cfg), Xva)
        b = C.predict_rescue_proba(C.fit_router(name, Xtr, ytr, ALLOWED, cfg), Xva)
        assert np.allclose(a, b), name


def test_unknown_model_and_config_errors(cfg):
    Xtr, ytr = make_xy()
    with pytest.raises(ValueError, match="unknown model"):
        C.fit_router("svm", Xtr, ytr, ALLOWED, cfg)
    with pytest.raises(FileNotFoundError):
        C.load_model_config("/nonexistent.yaml")


def test_xgboost_auto_scale_pos_weight_from_train(cfg):
    pytest.importorskip("xgboost")
    Xtr, ytr = make_xy()
    m = C.fit_router("xgboost", Xtr, ytr, ALLOWED, cfg)
    expected = (ytr == 0).sum() / (ytr == 1).sum()
    assert m.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_model_settings_recorded(cfg):
    s = C.model_settings(cfg)
    assert s["seed"] == 42 and set(s["models"]) == set(C.MODEL_NAMES)
