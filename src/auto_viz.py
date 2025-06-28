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
    st.title("模式识别交互式可视化平台")
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
    if st.button("训练并评估模型"):
        model = get_model(model_name)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        st.success(f"{model_name} 验证集准确率: {acc:.4f}")

    # 3. 特征重要性可解释性分析
    st.header("特征重要性可解释性分析")
    if st.button("训练并分析随机森林"):
        model = get_model("random_forest")
        model.fit(X_train, y_train)
        analyzer = SHAPAnalyzer(save_dir="results/shap")
        analyzer.explain(model, X_val, feature_names=list(X_train.columns), model_type="tree", plot_types=("bar", "summary"))
        st.image("results/shap/shap_bar.png", caption="特征重要性条形图")
        st.image("results/shap/shap_summary.png", caption="SHAP summary图")

if __name__ == "__main__":
    run_streamlit()
