"""
交互式可视化脚本（Streamlit）
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from src.models import get_model
from src.preprocess import load_data, preprocess_data
from src.shap_analyzer import SHAPAnalyzer

def prepare_data(test_size=0.2, random_state=42):
    df_raw = load_data()
    df = preprocess_data(df_raw)
    X = df.drop("label", axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_streamlit():
    st.title("模式识别")
    X_train, X_val, y_train, y_val = prepare_data()
    st.write("数据集大小：", X_train.shape[0] + X_val.shape[0])

    # 1. 数据分布可视化
    st.header("数据分布可视化")
    if st.checkbox("显示标签分布"):
        label_counts = pd.concat([y_train, y_val]).value_counts().sort_index()
        st.bar_chart(label_counts)
    if st.checkbox("显示特征分布"):
        feature = st.selectbox("选择特征", X_train.columns)
        st.bar_chart(X_train[feature].value_counts().sort_index())

    # 2. 训练与评估
    st.header("模型训练与评估")
    model_name = st.selectbox("选择模型", ["random_forest", "decision_tree", "knn", "svm", "mlp", "logistic_regression"])
    use_best = st.checkbox("使用最优超参数 (from best_params.json)")
    if use_best:
        try:
            import json
            with open("src/best_params.json", "r") as f:
                best_params = json.load(f)
            params = best_params.get(model_name, {})
        except Exception:
            params = {}
        st.write(f"当前参数: {params}")
        model = get_model(model_name, **params)
    else:
        model = get_model(model_name)
    if st.button("训练并评估模型"):
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        st.success(f"{model_name} 验证集准确率: {acc:.4f}")
    
    # 3. 超参数自动调参
    st.header("超参数自动调参 (Optuna)")
    tune_model = st.selectbox("选择模型进行调参", ["random_forest", "decision_tree", "knn", "svm", "mlp", "logistic_regression"])
    n_trials = st.slider("设置调参次数", min_value=5, max_value=100, value=20, step=5)
    if st.button("运行超参数调参"):
        from src.auto_optuna import optimize_random_forest, optimize_logistic_regression, optimize_knn, optimize_decision_tree, optimize_svm, optimize_mlp
        optimizers = {
            'random_forest': optimize_random_forest,
            'logistic_regression': optimize_logistic_regression,
            'knn': optimize_knn,
            'decision_tree': optimize_decision_tree,
            'svm': optimize_svm,
            'mlp': optimize_mlp,
        }
        with st.spinner('调参中，请稍候...'):
            opt_func = optimizers[tune_model]
            best, score = opt_func(X_train, X_val, y_train, y_val, n_trials=n_trials)
        st.write(f"最优参数: {best}")
        st.write(f"最佳验证集准确率: {score:.4f}")
        # 保存最优参数
        import json
        try:
            with open("src/best_params.json", "r") as f:
                all_best = json.load(f)
        except Exception:
            all_best = {}
        all_best[tune_model] = best
        with open("src/best_params.json", "w") as f:
            json.dump(all_best, f, indent=4)
        st.success("超参数已保存到 best_params.json")

    # 4. 特征重要性可解释性分析
    st.header("特征重要性可解释性分析")
    # 选择模型并可选最优参数
    model_shap = st.selectbox(
        "选择模型进行 SHAP 分析",
        ["random_forest", "decision_tree", "knn", "svm", "mlp", "logistic_regression"]
    )
    use_best_shap = st.checkbox("使用最优超参数 (from best_params.json)", key="shap_best")
    if use_best_shap:
        try:
            import json
            with open("src/best_params.json", "r") as f:
                best_params = json.load(f)
        except Exception:
            best_params = {}
        params_shap = best_params.get(model_shap, {})
        st.write(f"当前参数: {params_shap}")
        model = get_model(model_shap, **params_shap)
    else:
        model = get_model(model_shap)
    if st.button("训练并分析", key="shap_analyze"):
        model.fit(X_train, y_train)
        analyzer = SHAPAnalyzer(save_dir="results/shap")
        # 根据树模型或其他模型选择解释方式
        model_type = "tree" if model_shap in ("random_forest", "decision_tree") else "kernel"
        analyzer.explain(
            model,
            X_val,
            feature_names=list(X_train.columns),
            model_type=model_type,
            plot_types=("bar", "summary")
        )
        st.image("results/shap/shap_bar.png", caption="SHAP 条形图")
        st.image("results/shap/shap_beeswarm.png", caption="SHAP 蜂群图")
        # st.image("results/shap/shap_summary.png", caption="SHAP summary 图")

if __name__ == "__main__":
    run_streamlit()
