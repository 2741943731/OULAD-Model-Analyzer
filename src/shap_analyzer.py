"""
SHAP分析工具类：用于解释模型特征重要性并生成可视化图表
"""
import os
import shap
import matplotlib.pyplot as plt

class SHAPAnalyzer:
    def __init__(self, save_dir="results/shap"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def explain(self, model, X, feature_names=None, model_type="tree", plot_types=("bar", "summary")):
        """
        计算SHAP值并保存特征重要性图表
        参数：
            model: 已训练的模型
            X: 输入特征（numpy数组或DataFrame）
            feature_names: 特征名列表
            model_type: "tree"（树模型）、"linear"、"kernel"等
            plot_types: 生成的图表类型（"bar"、"summary"）
        """
        # 选择解释器
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        else:
            # 默认用KernelExplainer
            background = shap.sample(X, min(100, X.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X)

        # 处理特征名
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 生成并保存图表
        if "bar" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "shap_bar.png"))
            plt.close()
        if "summary" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "shap_summary.png"))
            plt.close()

        print(f"SHAP分析完成，图表已保存到: {self.save_dir}")
