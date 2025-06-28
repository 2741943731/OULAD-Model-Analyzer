"""
训练、评估与对比分析主流程
"""

from .models import get_model
from .utils import evaluate_model
import json

def train_and_evaluate(X_train, y_train, X_test, y_test, model_names, best_params=None):
    results = {}
    for name in model_names:
        print(f"处理模型: {name}")
        if best_params and name in best_params:
            print(f"  使用文档中{name}最优参数初始化...")
            model = get_model(name, **best_params[name])
        else:
            print(f"  使用默认参数初始化{name}...")
            model = get_model(name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
    return results
