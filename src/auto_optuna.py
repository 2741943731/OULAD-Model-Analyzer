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
import json

def prepare_data(test_size=0.2, random_state=42):
    df_raw = load_data()
    df = preprocess_data(df_raw)
    X = df.drop("label", axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ====== 随机森林参数优化 ======
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

# ====== 逻辑回归参数优化 ======
def optimize_logistic_regression(X_train, X_val, y_train, y_val, n_trials=20):
    def objective(trial):
        params = {
            'C': trial.suggest_loguniform('C', 1e-4, 1e2),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 100, 500, step=100),
        }
        model = get_model("logistic_regression", **params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# ====== KNN 参数优化 ======
def optimize_knn(X_train, X_val, y_train, y_val, n_trials=20):
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev']),
        }
        model = get_model("knn", **params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# ====== 决策树参数优化 ======
def optimize_decision_tree(X_train, X_val, y_train, y_val, n_trials=20):
    def objective(trial):
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        }
        model = get_model("decision_tree", **params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# ====== SVM 参数优化 ======
def optimize_svm(X_train, X_val, y_train, y_val, n_trials=5):
    # 抽样减少数据量以加快模型训练
    X_sub = X_train.sample(n=min(500, len(X_train)), random_state=42)
    y_sub = y_train.loc[X_sub.index]
    def objective(trial):
        # 第 trial.number +1 次
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'max_iter': trial.suggest_int('max_iter', 100, 300, step=100),
        }
        model = get_model("svm", **params)
        model.fit(X_sub, y_sub)
        score = model.score(X_val, y_val)
        return score
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# ====== MLP 参数优化 ======
def optimize_mlp(X_train, X_val, y_train, y_val, n_trials=20):
    def objective(trial):
        params = {
            'hidden_layer_sizes': (trial.suggest_int('n_units', 10, 100),),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 200, 1000, step=200),
        }
        model = get_model("mlp", **params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# ====== 更新 main: 遍历所有模型优化并保存结果 ======
def main():
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")
    # 暂不对 SVM 进行超参数优化，避免长时间卡住
    optimizers = [
        ("mlp", optimize_mlp),
        ("random_forest", optimize_random_forest),
        ("logistic_regression", optimize_logistic_regression),
        ("knn", optimize_knn)
    ]
    best_params_all = {}
    for name, func in optimizers:
        print(f"\nOptimizing {name}...")
        best_params, best_score = func(X_train, X_val, y_train, y_val)
        print(f"{name} best params: {best_params}, 验证集准确率: {best_score:.4f}")
        best_params_all[name] = best_params
    # 保存所有模型最优参数
    with open("best_params.json", "w") as f:
        json.dump(best_params_all, f, indent=4)
    print("Saved best parameters to best_params.json")

if __name__ == "__main__":
    main()
