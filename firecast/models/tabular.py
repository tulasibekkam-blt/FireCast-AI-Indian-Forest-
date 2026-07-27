from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model_factories(seed: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "svm": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, class_weight="balanced", random_state=seed))]),
        "mlp": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(max_iter=500, early_stopping=True, random_state=seed))]),
        "random_forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, random_state=seed),
    }
    optional = (("xgboost", "XGBClassifier"), ("lightgbm", "LGBMClassifier"), ("catboost", "CatBoostClassifier"))
    for name, class_name in optional:
        try:
            if name == "xgboost":
                from xgboost import XGBClassifier
                models[name] = XGBClassifier(n_estimators=400, eval_metric="logloss", random_state=seed)
            elif name == "lightgbm":
                from lightgbm import LGBMClassifier
                models[name] = LGBMClassifier(n_estimators=400, verbosity=-1, random_state=seed)
            else:
                from catboost import CatBoostClassifier
                models[name] = CatBoostClassifier(iterations=400, verbose=False, random_seed=seed)
        except ImportError:
            continue
    return models
