"""
主程序入口
"""

import os
from src.preprocess import load_data, preprocess_data
from src.train_eval import train_and_evaluate
from src.shap_analyzer import SHAPAnalyzer

import pandas as pd
import os
import sys

def main():
    # 加载与预处理（自动读取data/anonymisedData下所有OULAD表）
    try:
        df_raw = load_data()  # 默认统计前4周行为
    except Exception as e:
        print("数据加载失败，请检查data/anonymisedData目录和csv文件。错误信息：", e)
        return
    df = preprocess_data(df_raw)
    # 可选：保存处理后的数据，便于检查
    df.to_csv("processed_data.csv", index=False)
    print("已保存预处理后的数据到 processed_data.csv")
    # 特征与标签
    X = df.drop("label", axis=1)
    y = df["label"]
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 选择模型
    all_model_names = [
        "logistic_regression",
        "knn",
        "decision_tree",
        "svm",
        "random_forest",
        "mlp"
    ]
    # 支持命令行参数指定模型
    if len(sys.argv) > 1:
        model_names = [name for name in sys.argv[1:] if name in all_model_names]
        if not model_names:
            print(f"未识别的模型名，将运行全部模型。可选: {all_model_names}")
            model_names = all_model_names
    else:
        # 也可交互输入
        sel = input(f"请输入要运行的模型名（用逗号分隔，留空则全部）：\n可选: {all_model_names}\n").strip()
        if sel:
            model_names = [name.strip() for name in sel.split(',') if name.strip() in all_model_names]
            if not model_names:
                print("输入无效，将运行全部模型。")
                model_names = all_model_names
        else:
            model_names = all_model_names
    # 训练与评估
    # 从文件读取最优参数
    import json
    try:
        with open("src/best_params.json", "r") as f:
            best_params = json.load(f)
    except Exception:
        best_params = None
    results = train_and_evaluate(X_train, y_train, X_test, y_test, model_names, best_params=best_params)
    print("各模型评估结果：")
    for name, metrics in results.items():
        print(f"{name}: 准确率={metrics['accuracy']:.4f}")

    # # 针对每个模型做SHAP分析
    # analyzer = SHAPAnalyzer(save_dir="results/shap")
    # # 只裁剪样本数，不裁剪特征数，保证特征和训练一致
    # shap_sample_num = 10  # 分析前10个样本
    # shap_plot_types = ("bar", "summary", "interaction")  # 可选：bar, summary, interaction
    # for name in model_names:
    #     if name == "logistic_regression":
    #         continue
    #     model = None
    #     from src.models import get_model
    #     model = get_model(name)
    #     model.fit(X_train, y_train)
    #     # 判断模型类型
    #     if name in ["random_forest", "decision_tree"]:
    #         model_type = "tree"
    #     else:
    #         model_type = "kernel"
    #     # 只取前N个样本，特征全部保留
    #     shap_X = X_test.iloc[:shap_sample_num, :]
    #     shap_feature_names = list(X.columns)
    #     print(f"正在对{name}进行SHAP分析...（仅前{shap_sample_num}个样本，全部特征）")
    #     analyzer.explain(model, shap_X, feature_names=shap_feature_names, model_type=model_type, plot_types=shap_plot_types)

if __name__ == "__main__":
    main()
