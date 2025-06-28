"""
所有算法实现，每人负责2个算法
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# ====== 逻辑回归 ======
def logistic_regression_model(**kwargs):
    """
    直接返回sklearn的LogisticRegression实例
    参数可通过kwargs传递，如 penalty, C, solver, max_iter 等
    """
    return LogisticRegression(**kwargs)

# ====== KNN分类器 ======
def knn_model(**kwargs):
    """
    K近邻分类器
    参数可通过kwargs传递，如 n_neighbors, weights, metric 等
    返回：sklearn的KNeighborsClassifier实例
    """
    return KNeighborsClassifier(**kwargs)

# ====== 决策树分类器 ======
def decision_tree_model(**kwargs):
    """
    决策树分类器
    参数可通过kwargs传递，如 criterion, max_depth, min_samples_split 等
    返回：sklearn的DecisionTreeClassifier实例
    """
    return DecisionTreeClassifier(**kwargs)

# ====== 支持向量机 ======
def svm_model(**kwargs):
    """
    支持向量机分类器
    参数可通过kwargs传递，如 kernel, C, gamma 等
    返回：sklearn的SVC实例，默认probability=True以支持predict_proba和SHAP
    """
    kwargs.setdefault('probability', True)
    return SVC(**kwargs)

# ====== 随机森林分类器 ======
def random_forest_model(**kwargs):
    """
    随机森林分类器
    参数可通过kwargs传递，如 n_estimators, max_depth, random_state 等
    返回：sklearn的RandomForestClassifier实例
    """
    return RandomForestClassifier(**kwargs)

# ====== 多层感知机神经网络 ======
def mlp_model(**kwargs):
    """
    多层感知机（神经网络）分类器
    参数可通过kwargs传递，如 hidden_layer_sizes, activation, solver 等
    返回：sklearn的MLPClassifier实例
    """
    return MLPClassifier(**kwargs)


def get_model(name, **kwargs):
    """
    根据名称返回模型实例
    支持自定义参数传递
    """
    if name == "logistic_regression":
        return logistic_regression_model(**kwargs)
    elif name == "knn":
        return knn_model(**kwargs)
    elif name == "decision_tree":
        return decision_tree_model(**kwargs)
    elif name == "svm":
        return svm_model(**kwargs)
    elif name == "random_forest":
        return random_forest_model(**kwargs)
    elif name == "mlp":
        return mlp_model(**kwargs)
    else:
        raise ValueError(f"未知模型: {name}")

# 每个模型可单独写训练/预测函数，或统一用sklearn接口
