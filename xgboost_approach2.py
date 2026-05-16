from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
N_CV_SPLITS = 3
N_SEARCH_ITER = 24
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 2000


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


def sample_hyperparams(rng: np.random.Generator, n_iter: int) -> list[dict[str, Any]]:
    learning_rates = [0.01, 0.03, 0.05, 0.1]
    max_depths = [2, 3, 4, 5, 6]
    min_child_weights = [1, 2, 4, 8]
    subsamples = [0.5, 0.7, 0.85, 1.0]
    colsample_bytree_values = [0.4, 0.6, 0.8, 1.0]
    reg_lambdas = [0.0, 1.0, 5.0, 10.0]
    reg_alphas = [0.0, 0.1, 0.5, 1.0]
    gammas = [0.0, 0.5, 1.0]

    sampled: list[dict[str, Any]] = []

    for _ in range(n_iter):
        sampled.append(
            {
                "learning_rate": float(rng.choice(learning_rates)),
                "max_depth": int(rng.choice(max_depths)),
                "min_child_weight": float(rng.choice(min_child_weights)),
                "subsample": float(rng.choice(subsamples)),
                "colsample_bytree": float(rng.choice(colsample_bytree_values)),
                "reg_lambda": float(rng.choice(reg_lambdas)),
                "reg_alpha": float(rng.choice(reg_alphas)),
                "gamma": float(rng.choice(gammas)),
            }
        )

    return sampled


def make_xgb_model(params: dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=MAX_ESTIMATORS,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def train_with_internal_val(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> XGBClassifier:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    try:
        train_idx, val_idx = next(splitter.split(X, y))
    except ValueError:
        # Fallback for tiny folds; reuse train split as eval split to keep fit API consistent.
        train_idx = np.arange(X.shape[0])
        val_idx = np.arange(X.shape[0])

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = make_xgb_model(params, random_state=random_state)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_param_set_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=random_seed)

    fold_accuracy: list[float] = []
    fold_balanced_accuracy: list[float] = []
    fold_f1: list[float] = []
    fold_roc_auc: list[float] = []
    fold_best_iteration: list[int] = []

    for fold_idx, (fit_idx, eval_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit, X_eval = X_train[fit_idx], X_train[eval_idx]
        y_fit, y_eval = y_train[fit_idx], y_train[eval_idx]

        model = train_with_internal_val(
            X_fit,
            y_fit,
            params=params,
            random_state=random_seed + fold_idx,
        )

        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]

        fold_accuracy.append(float(accuracy_score(y_eval, y_pred)))
        fold_balanced_accuracy.append(float(balanced_accuracy_score(y_eval, y_pred)))
        fold_f1.append(float(f1_score(y_eval, y_pred, zero_division=0)))
        fold_roc_auc.append(safe_roc_auc(y_eval, y_proba))

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = MAX_ESTIMATORS - 1
        fold_best_iteration.append(int(best_iter) + 1)

    return {
        "cv_accuracy_mean": float(np.nanmean(fold_accuracy)),
        "cv_balanced_accuracy_mean": float(np.nanmean(fold_balanced_accuracy)),
        "cv_f1_mean": float(np.nanmean(fold_f1)),
        "cv_roc_auc_mean": float(np.nanmean(fold_roc_auc)),
        "cv_best_iteration_mean": float(np.nanmean(fold_best_iteration)),
    }


def run_task_search(task: TaskData, random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)

    X_train = task.X_train.to_numpy(dtype=np.float32)
    X_test = task.X_test.to_numpy(dtype=np.float32)
    y_train = task.y_train.to_numpy()
    y_test = task.y_test.to_numpy()

    sampled_params = sample_hyperparams(rng, N_SEARCH_ITER)

    rows: list[dict[str, Any]] = []

    for trial_idx, params in enumerate(sampled_params, start=1):
        cv_metrics = evaluate_param_set_cv(
            X_train=X_train,
            y_train=y_train,
            params=params,
            random_seed=random_seed + trial_idx,
        )

        row = {
            "task": task.name,
            "model": (
                f"xgb_lr={params['learning_rate']}_depth={params['max_depth']}"
                f"_sub={params['subsample']}_col={params['colsample_bytree']}"
                f"_lambda={params['reg_lambda']}_alpha={params['reg_alpha']}"
                f"_gamma={params['gamma']}_mcw={params['min_child_weight']}"
            ),
            **params,
            "n_estimators_max": MAX_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "features": int(X_train.shape[1]),
            **cv_metrics,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        by=["cv_balanced_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = results_df.iloc[0].to_dict()

    best_params = {
        "learning_rate": float(best_row["learning_rate"]),
        "max_depth": int(best_row["max_depth"]),
        "min_child_weight": float(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_lambda": float(best_row["reg_lambda"]),
        "reg_alpha": float(best_row["reg_alpha"]),
        "gamma": float(best_row["gamma"]),
    }

    final_model = train_with_internal_val(
        X_train,
        y_train,
        params=best_params,
        random_state=random_seed + 999,
    )

    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    best_iteration = getattr(final_model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = MAX_ESTIMATORS - 1

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
        **best_row,
        "best_iteration": int(best_iteration) + 1,
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "test_roc_auc": safe_roc_auc(y_test, y_test_proba),
        "nonzero_importance_features": int(importance_df.shape[0]),
    }

    # Save per-task outputs.
    task_results_path = task.task_dir / "xgboost_random_search_results.csv"
    results_df.to_csv(task_results_path, index=False)

    importance_path = task.task_dir / "xgboost_feature_importance_gain.csv"
    importance_df.to_csv(importance_path, index=False)

    # Plot 1: CV vs test balanced accuracy for each sampled model.
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(results_df.shape[0])
    ax.bar(x, results_df["cv_balanced_accuracy_mean"], label="CV balanced accuracy", alpha=0.75)
    best_idx = int(results_df.index[results_df["model"] == best_summary["model"]][0])
    ax.scatter(
        [best_idx],
        [best_summary["test_balanced_accuracy"]],
        color="black",
        marker="o",
        s=70,
        label="Test balanced accuracy (best model)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"m{i+1}" for i in x], rotation=60, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Sampled model")
    ax.set_title(f"XGBoost model search: {task.name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(task.task_dir / "xgboost_balanced_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Top gain importance features.
    if not importance_df.empty:
        top_gain = importance_df.head(20).sort_values("gain", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.barh(top_gain["feature"], top_gain["gain"], color="tab:green", alpha=0.85)
        ax2.set_xlabel("XGBoost feature importance (gain)")
        ax2.set_ylabel("Feature")
        ax2.set_title(f"Top gain features: {task.name}")
        ax2.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        fig2.savefig(task.task_dir / "xgboost_feature_importance_gain.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

    return results_df, best_summary


def main() -> None:
    task_data = load_task_data()

    all_results: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []

    for idx, task_name in enumerate(TASK_NAMES):
        task = task_data[task_name]
        print(f"Running XGBoost randomized search for {task_name}...")
        task_results_df, best_summary = run_task_search(task, random_seed=RANDOM_SEED + idx * 100)
        all_results.append(task_results_df)
        best_rows.append(best_summary)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = all_results_df.sort_values(
        by=["task", "cv_balanced_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    all_results_path = PROCESSED_DIR / "xgboost_random_search_results_all_tasks.csv"
    all_results_df.to_csv(all_results_path, index=False)

    best_models_df = pd.DataFrame(best_rows).sort_values(
        by=["task", "cv_balanced_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    best_models_path = PROCESSED_DIR / "xgboost_best_models_by_task.csv"
    best_models_df.to_csv(best_models_path, index=False)

    print("Saved:")
    print(f"- {all_results_path}")
    print(f"- {best_models_path}")


if __name__ == "__main__":
    main()
