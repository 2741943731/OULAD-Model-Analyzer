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

# ====== 自定义多层感知机分类器 ======
class CustomMLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=0.0001,
                learning_rate_init=0.001, max_iter=200, tol=1e-4, random_state=None, momentum=0.9, batch_size=32):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.momentum = momentum
        self.batch_size = batch_size

    def _init_weights(self, n_features):
        rng = np.random.RandomState(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.coefs_, self.intercepts_ = [], []
        for i in range(len(layer_sizes)-1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
            # Xavier initialization
            scale = np.sqrt(2.0/(in_dim + out_dim))
            w = rng.normal(loc=0.0, scale=scale, size=(in_dim, out_dim))
            b = np.zeros(out_dim, dtype=float)
            self.coefs_.append(w)
            self.intercepts_.append(b)
        # 初始化动量
        self.vel_coefs_ = [np.zeros_like(w) for w in self.coefs_]
        self.vel_intercepts_ = [np.zeros_like(b) for b in self.intercepts_]

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'logistic':
            return 1/(1+np.exp(-x))
        else:
            return x

    def _activate_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'logistic':
            sig = 1/(1+np.exp(-x))
            return sig * (1 - sig)
        else:
            return np.ones_like(x)

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float).reshape(-1,1)
        n_samples, n_features = X_mat.shape
        self._init_weights(n_features)
        loss_old = np.inf
        # 使用 mini-batch + momentum
        for iteration in range(self.max_iter):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_sh, y_sh = X_mat[idx], y_vec[idx]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_sh[start:end]; yb = y_sh[start:end]
                # forward
                activations, pre_acts = [xb], []
                for w, b in zip(self.coefs_, self.intercepts_):
                    z = activations[-1].dot(w) + b
                    pre_acts.append(z)
                    a = self._activate(z) if w is not self.coefs_[-1] else 1/(1+np.exp(-z))
                    activations.append(a)
                output = activations[-1]
                # backward
                delta = (output - yb)
                for i in reversed(range(len(self.coefs_))):
                    a_prev = activations[i]
                    dw = a_prev.T.dot(delta)/len(xb) + self.alpha*self.coefs_[i]
                    db = np.mean(delta, axis=0)
                    # momentum update
                    self.vel_coefs_[i] = self.momentum*self.vel_coefs_[i] - self.learning_rate_init*dw
                    self.vel_intercepts_[i] = self.momentum*self.vel_intercepts_[i] - self.learning_rate_init*db
                    self.coefs_[i] += self.vel_coefs_[i]
                    self.intercepts_[i] += self.vel_intercepts_[i]
                    if i > 0:
                        da = delta.dot(self.coefs_[i].T)
                        delta = da * self._activate_derivative(pre_acts[i-1])
            # 可选：计算整体 loss 检查收敛
            # early stop based on tol omitted for speed
        # 保存权重
        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=float)
        a = X_mat
        for w,b in zip(self.coefs_, self.intercepts_):
            z = a.dot(w) + b
            a = 1/(1+np.exp(-z)) if w is self.coefs_[-1] else self._activate(z)
        prob = a.ravel()
        return np.vstack([1-prob, prob]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)

# ====== 修改 mlp_model 返回自定义实现 ======
def mlp_model(**kwargs):
    """
    多层感知机（神经网络）分类器，使用自定义实现
    参数可通过kwargs传递，如 hidden_layer_sizes, activation, solver 等
    返回：CustomMLPClassifier实例
    """
    # return MLPClassifier(**kwargs)
    return CustomMLPClassifier(**kwargs)


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
