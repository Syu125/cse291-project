from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier


PROCESSED_DIR = Path("data/processed")
TASK_NAMES = [
    "healthy_vs_infected",
    "symptomatic_non_covid_vs_covid",
    "severe_vs_nonsevere",
]

RANDOM_SEED = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
N_SEARCH_ITER = 160
EARLY_STOPPING_ROUNDS = 50


@dataclass
class TaskData:
    name: str
    task_dir: Path
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    try:
        return float(fn(y_true, y_score, **kwargs))
    except Exception:
        return float("nan")


def load_task_data() -> dict[str, TaskData]:
    task_data: dict[str, TaskData] = {}

    for task_name in TASK_NAMES:
        task_dir = PROCESSED_DIR / task_name
        X_train = pd.read_csv(task_dir / "X_train_scaled.csv", index_col=0)
        X_test = pd.read_csv(task_dir / "X_test_scaled.csv", index_col=0)
        y_train = pd.read_csv(task_dir / "y_train.csv", index_col=0).squeeze("columns")
        y_test = pd.read_csv(task_dir / "y_test.csv", index_col=0).squeeze("columns")

        task_data[task_name] = TaskData(
            name=task_name,
            task_dir=task_dir,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    return task_data


def compute_scale_pos_weight_candidates(y: np.ndarray) -> list[float]:
    y_series = pd.Series(y)
    positives = float((y_series == 1).sum())
    negatives = float((y_series == 0).sum())
    if positives <= 0:
        return [1.0]

    base_ratio = max(1.0, negatives / positives)
    candidates = {
        1.0,
        round(base_ratio, 3),
        round(max(1.0, 0.5 * base_ratio), 3),
        round(min(20.0, 1.5 * base_ratio), 3),
    }
    return sorted(float(v) for v in candidates)


def sample_hyperparams(rng: np.random.Generator, n_iter: int, y: np.ndarray) -> list[dict[str, Any]]:
    n_estimators_values = [500, 1000, 1500, 2000]
    learning_rates = [0.005, 0.01, 0.03, 0.05]
    max_depths = [2, 3, 4, 5]
    min_child_weights = [1, 2, 4, 8]
    subsamples = [0.5, 0.7, 0.85, 1.0]
    colsample_bytree_values = [0.3, 0.5, 0.7, 1.0]
    reg_lambdas = [1.0, 5.0, 10.0, 20.0]
    reg_alphas = [0.0, 0.1, 0.5, 1.0, 5.0]
    gammas = [0.0, 0.5, 1.0, 2.0]
    scale_pos_weight_values = compute_scale_pos_weight_candidates(y)

    sampled: list[dict[str, Any]] = []
    for _ in range(n_iter):
        sampled.append(
            {
                "n_estimators": int(rng.choice(n_estimators_values)),
                "learning_rate": float(rng.choice(learning_rates)),
                "max_depth": int(rng.choice(max_depths)),
                "min_child_weight": float(rng.choice(min_child_weights)),
                "subsample": float(rng.choice(subsamples)),
                "colsample_bytree": float(rng.choice(colsample_bytree_values)),
                "reg_lambda": float(rng.choice(reg_lambdas)),
                "reg_alpha": float(rng.choice(reg_alphas)),
                "gamma": float(rng.choice(gammas)),
                "scale_pos_weight": float(rng.choice(scale_pos_weight_values)),
            }
        )

    return sampled


def make_xgb_model(params: dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        scale_pos_weight=params["scale_pos_weight"],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def train_with_early_stopping(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> XGBClassifier:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    try:
        train_idx, val_idx = next(splitter.split(X, y))
    except ValueError:
        train_idx = np.arange(X.shape[0])
        val_idx = np.arange(X.shape[0])

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = make_xgb_model(params, random_state=random_state)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_nested_cv(
    task: TaskData,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=random_seed)
    inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_seed)

    X_train = task.X_train.to_numpy(dtype=np.float32)
    y_train = task.y_train.to_numpy()

    rng = np.random.default_rng(random_seed)
    sampled_params = sample_hyperparams(rng, N_SEARCH_ITER, y_train)

    outer_scores: list[dict[str, Any]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
        X_outer_train, X_outer_val = X_train[train_idx], X_train[val_idx]
        y_outer_train, y_outer_val = y_train[train_idx], y_train[val_idx]

        best_inner_score = -np.inf
        best_params = sampled_params[0]

        for trial_idx, params in enumerate(sampled_params):
            inner_fold_scores: list[float] = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
                X_in = X_outer_train[inner_train_idx]
                X_iv = X_outer_train[inner_val_idx]
                y_in = y_outer_train[inner_train_idx]
                y_iv = y_outer_train[inner_val_idx]

                model = train_with_early_stopping(
                    X_in,
                    y_in,
                    params,
                    random_state=random_seed + fold_idx * 1000 + trial_idx * 10,
                )
                y_pred_iv = model.predict(X_iv)
                inner_fold_scores.append(float(balanced_accuracy_score(y_iv, y_pred_iv)))

            mean_inner = float(np.mean(inner_fold_scores))
            if mean_inner > best_inner_score:
                best_inner_score = mean_inner
                best_params = params

        best_model = train_with_early_stopping(
            X_outer_train,
            y_outer_train,
            best_params,
            random_state=random_seed + fold_idx * 10000 + 999,
        )
        y_proba = best_model.predict_proba(X_outer_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        fold_result = {
            "fold": fold_idx,
            **best_params,
            "val_accuracy": float(accuracy_score(y_outer_val, y_pred)),
            "val_balanced_accuracy": float(balanced_accuracy_score(y_outer_val, y_pred)),
            "val_f1": float(f1_score(y_outer_val, y_pred, zero_division=0)),
            "val_roc_auc": safe_roc_auc(y_outer_val, y_proba),
            "val_auprc": safe_metric(average_precision_score, y_outer_val, y_proba),
        }
        outer_scores.append(fold_result)
        best_params_per_fold.append(best_params)

        print(
            f"  Fold {fold_idx}: "
            f"n_estimators={best_params['n_estimators']}, "
            f"lr={best_params['learning_rate']}, "
            f"depth={best_params['max_depth']} | "
            f"bal_acc={fold_result['val_balanced_accuracy']:.3f}, "
            f"AUPRC={fold_result['val_auprc']:.3f}"
        )

    return outer_scores, best_params_per_fold


def run_task_search(task: TaskData, random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    X_train = task.X_train.to_numpy(dtype=np.float32)
    X_test = task.X_test.to_numpy(dtype=np.float32)
    y_train = task.y_train.to_numpy()
    y_test = task.y_test.to_numpy()

    outer_scores, _ = run_nested_cv(task, random_seed=random_seed)

    aggregate_cv = {
        "cv_accuracy_mean": float(np.mean([r["val_accuracy"] for r in outer_scores])),
        "cv_balanced_accuracy_mean": float(np.mean([r["val_balanced_accuracy"] for r in outer_scores])),
        "cv_f1_mean": float(np.mean([r["val_f1"] for r in outer_scores])),
        "cv_roc_auc_mean": float(np.nanmean([r["val_roc_auc"] for r in outer_scores])),
        "cv_auprc_mean": float(np.nanmean([r["val_auprc"] for r in outer_scores])),
    }

    inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_seed)
    rng = np.random.default_rng(random_seed + 1)
    sampled_params = sample_hyperparams(rng, N_SEARCH_ITER, y_train)

    search_rows: list[dict[str, Any]] = []
    best_inner_score = -np.inf
    best_params: dict[str, Any] = sampled_params[0]

    for trial_idx, params in enumerate(sampled_params, start=1):
        fold_scores: list[float] = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            X_in, X_iv = X_train[inner_train_idx], X_train[inner_val_idx]
            y_in, y_iv = y_train[inner_train_idx], y_train[inner_val_idx]
            model = train_with_early_stopping(
                X_in,
                y_in,
                params,
                random_state=random_seed + trial_idx * 10,
            )
            y_pred_iv = model.predict(X_iv)
            fold_scores.append(float(balanced_accuracy_score(y_iv, y_pred_iv)))

        mean_score = float(np.mean(fold_scores))
        search_rows.append({
            "task": task.name,
            **params,
            "inner_cv_balanced_accuracy_mean": mean_score,
        })

        if mean_score > best_inner_score:
            best_inner_score = mean_score
            best_params = params

    search_df = pd.DataFrame(search_rows).sort_values(
        "inner_cv_balanced_accuracy_mean", ascending=False
    ).reset_index(drop=True)

    final_model = train_with_early_stopping(
        X_train,
        y_train,
        params=best_params,
        random_state=random_seed + 999,
    )

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    best_iteration = getattr(final_model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = best_params["n_estimators"] - 1

    booster = final_model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")

    importance_rows: list[dict[str, Any]] = []
    for feat, gain in gain_scores.items():
        feat_idx = int(feat[1:])
        importance_rows.append(
            {
                "task": task.name,
                "feature": task.X_train.columns[feat_idx],
                "gain": float(gain),
            }
        )

    importance_df = pd.DataFrame(importance_rows).sort_values("gain", ascending=False).reset_index(drop=True)

    best_summary = {
        "task": task.name,
        **best_params,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "decision_threshold": 0.5,
        "best_iteration": int(best_iteration) + 1,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": int(X_train.shape[1]),
        **aggregate_cv,
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "test_roc_auc": safe_roc_auc(y_test, y_test_proba),
        "test_auprc": safe_metric(average_precision_score, y_test, y_test_proba),
        "nonzero_importance_features": int(importance_df.shape[0]),
    }

    fold_df = pd.DataFrame(outer_scores)
    fold_df.to_csv(task.task_dir / "xgboost_nonlinear_nested_cv_folds.csv", index=False)
    search_df.to_csv(task.task_dir / "xgboost_nonlinear_random_search_results.csv", index=False)
    importance_df.to_csv(task.task_dir / "xgboost_nonlinear_feature_importance_gain.csv", index=False)

    return search_df, best_summary


def main() -> None:
    task_data = load_task_data()

    all_results: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []

    for idx, task_name in enumerate(TASK_NAMES):
        task = task_data[task_name]
        print(f"Running XGBoost nonlinear model for {task_name}...")
        task_results_df, best_summary = run_task_search(task, random_seed=RANDOM_SEED + idx * 100)
        all_results.append(task_results_df)
        best_rows.append(best_summary)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = all_results_df.sort_values(
        by=["task", "inner_cv_balanced_accuracy_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    all_results_path = PROCESSED_DIR / "xgboost_nonlinear_random_search_results_all_tasks.csv"
    all_results_df.to_csv(all_results_path, index=False)

    best_models_df = pd.DataFrame(best_rows).sort_values(
        by=["task", "cv_balanced_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    best_models_path = PROCESSED_DIR / "xgboost_nonlinear_final_results_by_task.csv"
    best_models_df.to_csv(best_models_path, index=False)

    print("Saved:")
    print(f"- {all_results_path}")
    print(f"- {best_models_path}")


if __name__ == "__main__":
    main()
