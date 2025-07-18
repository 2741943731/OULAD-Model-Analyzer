"""
SHAP分析工具类：用于解释模型特征重要性并生成可视化图表
"""
import os
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback

class SHAPAnalyzer:
    def __init__(self, save_dir="results/shap"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def explain(self, model, X, feature_names=None, model_type="tree", plot_types=("bar", "summary", "dependence")):
        """
        计算SHAP值并生成多种可视化图表
        """
        try:
            print(f"Starting SHAP analysis: model_type={model_type}, data shape={getattr(X, 'shape', None)}")
            # 检查是否是自定义模型
            model_class_name = type(model).__name__
            # 选择 SHAP 解释器
            if model_type == "tree" and "Custom" not in model_class_name:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                # 默认 KernelExplainer，并抽样背景数据
                bg_size = min(100, X.shape[0])
                background = shap.sample(X, bg_size)
                # test_pred = model.predict_proba(background[:1])
                explainer = shap.KernelExplainer(model.predict, background)
                sample_size = min(50, X.shape[0])
                X_sample = shap.sample(X, sample_size)
                shap_values = explainer.shap_values(X_sample)
                X = X_sample
                
            # 统一 shap_values 到二维数组
            if isinstance(shap_values, list):
                # 二分类取第二个元素，否则取第一个
                shap_arr = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                shap_arr = shap_values
            # 展平高维 SHAP 输出
            if shap_arr.ndim > 2:
                shap_arr = shap_arr.reshape(shap_arr.shape[0], -1)

            # 特征名处理
            if feature_names is None:
                if hasattr(X, 'columns'):
                    feature_names = list(X.columns)
                else:
                    feature_names = [f'feature_{i}' for i in range(shap_arr.shape[1])]

            # 绘制并保存图表
            self.generate_plots(shap_arr, X, feature_names, plot_types)
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            traceback.print_exc()

    def generate_plots(self, shap_values, X, feature_names, plot_types):
        # 确保 shape 对齐
        if shap_values.shape[0] != X.shape[0]:
            raise ValueError(f"样本数不匹配: shap {shap_values.shape[0]} vs X {X.shape[0]}")
        
        # 确保特征数匹配
        if shap_values.shape[1] != X.shape[1]:
            print(f"Feature mismatch: truncating to min({shap_values.shape[1]}, {X.shape[1]})")
            min_features = min(shap_values.shape[1], X.shape[1])
            shap_values = shap_values[:, :min_features]
            if hasattr(X, 'iloc'):
                X = X.iloc[:, :min_features]
            else:
                X = X[:, :min_features]
            feature_names = feature_names[:min_features]

        # 条形图
        if "bar" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            plt.savefig(os.path.join(self.save_dir, "shap_bar.png"))
            plt.close()
        # 蜂群图
        if "summary" in plot_types or "beeswarm" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            # 增加特征间垂直间距：根据特征数量调整图形高度，并扩展宽度以显示完整标签
            fig = plt.gcf()
            num_features = shap_values.shape[1]
            fig.set_size_inches(10, max(6, 0.5 * num_features))  # 扩大宽度
            plt.subplots_adjust(left=0.35)  # 增加左侧边距
            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(self.save_dir, "shap_beeswarm.png"))
            plt.close()
        # 特征依赖图
        if "dependence" in plot_types:
            for idx, feat in enumerate(feature_names):
                plt.figure()
                shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
                plt.savefig(os.path.join(self.save_dir, f"dependence_{feat}.png"))
                plt.close()

        print(f"SHAP分析完成，图表已保存到: {self.save_dir}")
    
    def analyze_model(self, model, X, feature_names, model_name, label_type='label', plot_types=('bar','beeswarm','dependence')):
        """
        根据model_name自动选择SHAP解释器，生成对应目录下的图表并返回文件路径列表
        """
        # 设置专属输出目录
        out_dir = os.path.join(self.save_dir, label_type, model_name)
        os.makedirs(out_dir, exist_ok=True)
        # 选Explainer
        if model_name in ('random_forest','decision_tree'):
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X)
        elif model_name in ('mlp','lstm','rnn'):
            # 需将数据转为Tensor
            import torch
            X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
            expl = shap.DeepExplainer(model, X_t)
            sv = expl.shap_values(X_t)
        else:
            bg = shap.sample(X, min(100, X.shape[0]))
            expl = shap.KernelExplainer(model.predict_proba, bg)
            sv = expl.shap_values(X)
        # 提取正类SHAP二维数组
        if isinstance(sv, list) and len(sv) > 1:
            shap_arr = sv[1]
        else:
            shap_arr = sv if not isinstance(sv, list) else sv[0]
        if shap_arr.ndim > 2:
            shap_arr = shap_arr.reshape(shap_arr.shape[0], -1)
        # 使用专属目录暂存save_dir，调用generate_plots
        old = self.save_dir
        self.save_dir = out_dir
        self.generate_plots(shap_arr, X, feature_names, plot_types)
        self.save_dir = old
        # 返回生成文件
        files = []
        for p in os.listdir(out_dir):
            files.append(os.path.join(out_dir, p))
        return files
