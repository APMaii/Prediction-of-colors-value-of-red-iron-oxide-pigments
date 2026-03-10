"""
Learning curve analysis for the best SVR models (L*, a*, b*).

This script uses the exact best configurations from:
Backup/Practcial Main_Feb2026.py (lines 414-541)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.model_selection import KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR


def build_best_model(target_name: str):
    """Build the exact best model configuration for each target."""
    if target_name == "L*":
        # Best config from the provided block
        input_scaler = StandardScaler()
        svr = SVR(C=100, epsilon=0.3, gamma="scale", kernel="rbf")
        cv = KFold(n_splits=7, shuffle=True, random_state=42)
    elif target_name == "a*":
        # Best config from the provided block
        input_scaler = None
        svr = SVR(C=60, epsilon=0.3, gamma="scale", kernel="poly")
        cv = KFold(n_splits=7, shuffle=True, random_state=20)
    elif target_name == "b*":
        # Best config from the provided block
        input_scaler = StandardScaler()
        svr = SVR(C=9, epsilon=0.3, gamma="scale", kernel="rbf")
        cv = KFold(n_splits=7, shuffle=True, random_state=16)
    else:
        raise ValueError(f"Unsupported target: {target_name}")

    # Keep the original pipeline shape from your training code:
    # Pipeline([("scaler", MinMaxScaler()), ("model", SVR())])
    # with scaler overridden by best config (None or StandardScaler()).
    pipe_model = Pipeline([("scaler", MinMaxScaler()), ("model", svr)])
    pipe_model.set_params(scaler=input_scaler)

    model = TTR(regressor=pipe_model, transformer=StandardScaler())
    return model, cv


def run_learning_curve(X: np.ndarray, y: np.ndarray, target_name: str, out_dir: Path):
    """Compute, plot, and save learning curve outputs for one target."""
    model, cv = build_best_model(target_name)
    train_sizes = np.linspace(0.2, 1.0, 6)

    sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y.ravel(),
        cv=cv,
        train_sizes=train_sizes,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    train_rmse = -train_scores
    val_rmse = -val_scores

    # Save curve data for later reporting/table use.
    curve_df = pd.DataFrame(
        {
            "target": target_name,
            "train_size": sizes,
            "train_rmse_mean": train_rmse.mean(axis=1),
            "train_rmse_std": train_rmse.std(axis=1),
            "val_rmse_mean": val_rmse.mean(axis=1),
            "val_rmse_std": val_rmse.std(axis=1),
            "generalization_gap": val_rmse.mean(axis=1) - train_rmse.mean(axis=1),
        }
    )
    curve_df.to_csv(out_dir / f"learning_curve_{target_name.replace('*', 'star')}.csv", index=False)

    # Plot (mean +/- std).
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, train_rmse.mean(axis=1), "o-", linewidth=2, label="Train RMSE")
    plt.plot(sizes, val_rmse.mean(axis=1), "o-", linewidth=2, label="Validation RMSE")

    plt.fill_between(
        sizes,
        train_rmse.mean(axis=1) - train_rmse.std(axis=1),
        train_rmse.mean(axis=1) + train_rmse.std(axis=1),
        alpha=0.15,
    )
    plt.fill_between(
        sizes,
        val_rmse.mean(axis=1) - val_rmse.std(axis=1),
        val_rmse.mean(axis=1) + val_rmse.std(axis=1),
        alpha=0.15,
    )

    plt.title(f"Learning Curve ({target_name}) - Best SVR", fontsize=12)
    plt.xlabel("Training Samples")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"learning_curve_{target_name.replace('*', 'star')}.png", dpi=300)
    plt.close()

    return curve_df


def main():
    data = pd.read_excel("Experimental_data.xlsx")
    X = data[["Redox Agent", "Temperature", "Time"]].to_numpy()

    out_dir = Path("learning_curves")
    out_dir.mkdir(exist_ok=True)

    all_curves = []
    for target_name in ["L*", "a*", "b*"]:
        y = data[[target_name]].to_numpy()
        curve_df = run_learning_curve(X, y, target_name, out_dir)
        all_curves.append(curve_df)

        first_val = curve_df["val_rmse_mean"].iloc[0]
        last_val = curve_df["val_rmse_mean"].iloc[-1]
        improvement = ((first_val - last_val) / first_val) * 100
        print(
            f"{target_name}: validation RMSE improvement from smallest to largest "
            f"training size = {improvement:.2f}%"
        )

    summary_df = pd.concat(all_curves, ignore_index=True)
    summary_df.to_csv(out_dir / "learning_curve_summary_all_targets.csv", index=False)
    print(f"Saved learning-curve outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
