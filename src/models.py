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
import random
from collections import Counter

class CustomDecisionTree:
    """
    自定义决策树分类器实现
    支持关键参数调整：criterion, max_depth, min_samples_split, min_samples_leaf
    """
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # 避免log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _impurity(self, y):
        """根据criterion计算不纯度"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """计算信息增益"""
        parent_impurity = self._impurity(y)
        
        # 分割数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        y_left, y_right = y[left_mask], y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        # 计算加权不纯度
        n = len(y)
        weighted_impurity = (len(y_left) / n) * self._impurity(y_left) + \
                           (len(y_right) / n) * self._impurity(y_right)
        
        return parent_impurity - weighted_impurity
    
    def _best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # 获取该特征的唯一值作为候选阈值
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件
        n_samples = len(y)
        
        if (n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            # 返回叶节点
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value, 'samples': n_samples}
        
        # 找到最佳分割
        feature_idx, threshold, gain = self._best_split(X, y)
        
        if feature_idx is None or gain == 0:
            # 无法分割，返回叶节点
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value, 'samples': n_samples}
        
        # 分割数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # 检查最小叶子节点样本数
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value, 'samples': n_samples}
        
        # 递归构建子树
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples,
            'gain': gain
        }
    
    def fit(self, X, y):
        """训练决策树"""
        X = np.array(X)
        y = np.array(y)
        
        self.tree = self._build_tree(X, y)
        self._calculate_feature_importances(X)
        return self
    
    def _calculate_feature_importances(self, X):
        """计算特征重要性"""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        def traverse(node, n_samples):
            if node['leaf']:
                return
            
            feature_idx = node['feature']
            gain = node['gain']
            samples = node['samples']
            
            # 特征重要性 = (样本数 / 总样本数) * 信息增益
            importances[feature_idx] += (samples / n_samples) * gain
            
            traverse(node['left'], n_samples)
            traverse(node['right'], n_samples)
        
        traverse(self.tree, X.shape[0])
        
        # 归一化
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, sample):
        """预测单个样本"""
        node = self.tree
        
        while not node['leaf']:
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['value']
    
    def predict(self, X):
        """预测多个样本"""
        X = np.array(X)
        return np.array([self._predict_sample(sample) for sample in X])
    
    def predict_proba(self, X):
        """预测概率（简化实现）"""
        predictions = self.predict(X)
        unique_classes = np.unique(predictions)
        
        # 简化：直接返回0或1的概率
        probas = np.zeros((len(predictions), len(unique_classes)))
        for i, pred in enumerate(predictions):
            class_idx = np.where(unique_classes == pred)[0][0]
            probas[i, class_idx] = 1.0
        
        return probas
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    
class CustomRandomForest:
    """
    自定义随机森林分类器实现
    支持关键参数调整：n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Bootstrap抽样"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features):
        """获取每棵树使用的最大特征数"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """训练随机森林"""
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap抽样
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # 随机选择特征
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_subset = X_bootstrap[:, feature_indices]
            
            # 训练决策树
            tree = CustomDecisionTree(
                criterion='gini',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_subset, y_bootstrap)
            
            # 保存树和对应的特征索引
            self.trees.append({
                'tree': tree,
                'features': feature_indices
            })
        
        # 计算特征重要性
        self._calculate_feature_importances(n_features)
        
        return self
    
    def _calculate_feature_importances(self, n_features):
        """计算特征重要性"""
        importances = np.zeros(n_features)
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            features = tree_info['features']
            
            # 将树的特征重要性映射回原始特征空间
            for i, feature_idx in enumerate(features):
                importances[feature_idx] += tree.feature_importances_[i]
        
        # 归一化
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def predict(self, X):
        """预测多个样本"""
        X = np.array(X)
        predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            features = tree_info['features']
            
            # 使用相同的特征子集进行预测
            X_subset = X[:, features]
            tree_predictions = tree.predict(X_subset)
            predictions.append(tree_predictions)
        
        # 投票决定最终预测
        predictions = np.array(predictions).T
        final_predictions = []
        
        for sample_predictions in predictions:
            vote_counts = Counter(sample_predictions)
            final_predictions.append(vote_counts.most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.array(X)
        all_predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            features = tree_info['features']
            
            X_subset = X[:, features]
            tree_predictions = tree.predict(X_subset)
            all_predictions.append(tree_predictions)
        
        # 计算每个类别的投票比例
        all_predictions = np.array(all_predictions).T
        unique_classes = np.unique(np.concatenate(all_predictions))
        
        probas = np.zeros((len(X), len(unique_classes)))
        
        for i, sample_predictions in enumerate(all_predictions):
            vote_counts = Counter(sample_predictions)
            total_votes = len(sample_predictions)
            
            for j, class_label in enumerate(unique_classes):
                probas[i, j] = vote_counts.get(class_label, 0) / total_votes
        
        return probas
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    
# ====== 决策树分类器 ======
def decision_tree_model(**kwargs):
    """
    决策树分类器 - 支持自定义实现和sklearn实现
    默认使用自定义实现
    """
    # 可以通过参数选择使用自定义实现还是sklearn实现
    use_custom = kwargs.pop('use_custom', True)
    
    if use_custom:
        return CustomDecisionTree(**kwargs)
    else:
        return DecisionTreeClassifier(**kwargs)

# ====== 随机森林分类器 ======
def random_forest_model(**kwargs):
    """
    随机森林分类器 - 支持自定义实现和sklearn实现streamlit run auto_viz.py
    默认使用自定义实现
    """
    # 可以通过参数选择使用自定义实现还是sklearn实现
    use_custom = kwargs.pop('use_custom', True)
    
    if use_custom:
        return CustomRandomForest(**kwargs)
    else:
        return RandomForestClassifier(**kwargs)

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
