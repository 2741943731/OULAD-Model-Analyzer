"""
自动化调参脚本（Optuna）
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import get_model
from src.preprocess import load_data, preprocess_data

def prepare_data(test_size=0.2, random_state=42):
    df_raw = load_data()
    df = preprocess_data(df_raw)
    X = df.drop("label", axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def optimize_random_forest(X_train, X_val, y_train, y_val, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        }
        model = get_model("random_forest", **params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def main():
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")
    best_params, best_score = optimize_random_forest(X_train, X_val, y_train, y_val)
    print(f"最优参数: {best_params}, 验证集准确率: {best_score:.4f}")

if __name__ == "__main__":
    main()
