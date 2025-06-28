"""
训练、评估与对比分析主流程
"""

from .models import get_model
from .utils import evaluate_model

def train_and_evaluate(X_train, y_train, X_test, y_test, model_names):
    results = {}
    for name in model_names:
        print(f"训练模型: {name}")
        model = get_model(name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
    return results
