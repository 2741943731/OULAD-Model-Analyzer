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

class CustomKnn:
    def __init__(self, n_neighbors = 10, weights = "distance", metric = "manhattan"):
        """
        参数:
        n_neighbors: 邻居个数
        weights: uniform(多数投票) 或 distance(距离权重)
        metric: 距离类型，euclidean (欧式距离), manhattan (曼哈顿距离)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

        self._metric_functions = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'chebyshev': self._chebyshev_distance
        }

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes = np.unique(y)
        return self

    def predict_proba(self, X):
        """
        返回测试数据的概率估计
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        print(self.metric,self.n_neighbors,self.weights)

        if callable(self.metric):
            dist_func = self.metric
        else:
            dist_func = self._metric_functions.get(self.metric)
            if dist_func is None:
                raise ValueError(f"不支持的度量方法: {self.metric}")

        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            # 计算当前测试样本与所有训练样本的距离
            distances = dist_func(self.X_train, X[i])
            # 最近的k个邻居
            k_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]
            # 权重
            if callable(self.weights):
                weights = self.weights(k_distances)
            elif self.weights == 'distance':
                with np.errstate(divide='ignore'):
                    weights = 1.0 / (k_distances + 1e-10)
            else:  # uniform
                weights = np.ones_like(k_distances)

            class_weights = np.zeros(n_classes)
            for cls_idx, cls in enumerate(self.classes):
                mask = (k_labels == cls)
                class_weights[cls_idx] = np.sum(weights[mask])

            # 归一化得到概率
            total_weight = np.sum(class_weights)
            if total_weight > 0:
                proba[i] = class_weights / total_weight
            else:
                proba[i] = np.ones(n_classes) / n_classes  # 均匀分布

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes.take(np.argmax(proba, axis=1), axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # ===== 距离度量函数 =====
    @staticmethod
    def _euclidean_distance(X, x):
        """欧氏距离"""
        return np.sqrt(np.sum((X - x) ** 2, axis=1))

    @staticmethod
    def _manhattan_distance(X, x):
        """曼哈顿距离"""
        return np.sum(np.abs(X - x), axis=1)
    @staticmethod
    def _chebyshev_distance(X, x):
        """切比雪夫距离"""
        return np.max(np.abs(X - x), axis=1)

class CustomSVM:
    def __init__(self, C=1.0, lr=0.01, max_iter=1000, tol=1e-4, random_state=None, probability =True):
        """
        C : 正则化参数，默认为1
        learning_rate : 学习率，默认为0.01
        max_iter : 最大迭代次数.默认为1000
        tol : 损失变化容忍度（当损失变化小于tol时停止训练），默认为1e-4
        random_state : 随机种子，默认为NONE
        """
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def _compute_gradients(self, w, b, X, y):
        """计算权重和偏置的梯度"""
        n_samples = X.shape[0]
        gradients_w = np.zeros_like(w)
        gradient_b = 0.0
        loss = 0.0

        for i in range(n_samples):
            margin = y[i] * (np.dot(w, X[i]) + b)

            # 计算Hinge Loss和梯度
            if margin < 1:
                gradients_w += -y[i] * X[i]
                gradient_b += -y[i]
                loss += 1 - margin

        # 添加正则化项
        gradients_w = w + self.C * gradients_w
        loss = 0.5 * np.dot(w, w) + self.C * loss

        return gradients_w, gradient_b, loss

    def fit(self, X, y):
        """训练二分类SVM"""
        # 输入校验
        X = np.array(X)
        y = np.array(y)

        # 确保是二分类问题
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("此实现仅支持二分类问题")

        # 将标签转换为+1和-1
        y_transformed = np.where(y == self.classes_[0], -1, 1)
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(scale=0.01, size=n_features)
        b = 0.0

        # 梯度下降优化
        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # 计算梯度
            grad_w, grad_b, loss = self._compute_gradients(w, b, X, y_transformed)

            # 更新参数
            w -= self.lr * grad_w
            b -= self.lr * grad_b * 0.01  # 偏置使用较小的学习率

            # 检查收敛
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        # 保存模型参数
        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])

        return self

    def decision_function(self, X):
        """返回样本到决策边界的符号距离"""
        X = np.array(X)
        return (X @ self.coef_.T + self.intercept_).ravel()

    def predict(self, X):
        """预测样本类别"""
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """预测样本属于正类的概率（使用决策函数的logistic转换）"""
        scores = self.decision_function(X)
        proba = 1 / (1 + np.exp(-scores))
        return np.vstack((1 - proba, proba)).T

    def score(self, X, y):
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
    return CustomKnn(**kwargs)
    #return KNeighborsClassifier(**kwargs)

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

    return CustomSVM(**kwargs)
    #return SVC(**kwargs)

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
