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
class CustomLogisticRegression:
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=100, tol=1e-4, learning_rate=0.1):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def _sigmoid(self, z):
        # numerically stable sigmoid
        return 0.5 * (1 + np.tanh(0.5 * z))

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float)
        n_samples, n_features = X_mat.shape
        # 添加截距项
        X_bias = np.hstack([np.ones((n_samples, 1), dtype=float), X_mat])
        # 初始化权重
        w = np.zeros(n_features + 1)
        for i in range(self.max_iter):
            z = X_bias.dot(w)
            p = self._sigmoid(z)
            # 梯度计算，加入 L2 正则
            grad = (X_bias.T.dot(y_vec - p) - w / self.C) / float(n_samples)
            w_new = w + self.learning_rate * grad
            if np.linalg.norm(w_new - w, ord=1) < self.tol:
                w = w_new
                break
            w = w_new
        # 保存结果
        self.coef_ = w[1:].reshape(1, -1).astype(float)
        self.intercept_ = w[0]
        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=float)
        z = X_mat.dot(self.coef_.T) + self.intercept_
        p = self._sigmoid(z)
        return np.hstack([(1 - p), p])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """
        计算准确率，与 sklearn 接口一致
        """
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)

def logistic_regression_model(**kwargs):
    """
    自定义实现逻辑回归，参数与 sklearn 接口一致
    参数可通过 kwargs 传递，如 penalty, C, solver, max_iter 等
    """
    return CustomLogisticRegression(**kwargs)
    # return LogisticRegression(**kwargs)

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
