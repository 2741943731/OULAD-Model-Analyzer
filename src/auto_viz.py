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
    
    # 初始化 session_state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
        
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
        with st.spinner(f'正在训练 {model_name} 模型，请稍候...'):
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
        st.success(f"{model_name} 验证集准确率: {acc:.4f}")
        
        # 保存训练好的模型到 session_state
        st.session_state.trained_models[model_name] = {
            'model': model,
            'accuracy': acc,
            'params': params if use_best else {}
        }
        st.info(f"模型 {model_name} 已保存，可在特征重要性分析中使用")
    
    # 显示已训练的模型
    if st.session_state.trained_models:
        st.subheader("已训练的模型")
        for name, info in st.session_state.trained_models.items():
            st.write(f"- {name}: 准确率 {info['accuracy']:.4f}")
    
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

        # 模型选择方式
    if not st.session_state.trained_models:
        st.warning("没有已训练的模型，请先训练模型")
    else:
        model_keys = list(st.session_state.trained_models.keys())
        selected_model_key = st.selectbox("已训练的模型", model_keys)
            
        if st.button("分析特征重要性"):
            model_info = st.session_state.trained_models[selected_model_key]
            model = model_info['model']
            model_name = selected_model_key.replace('_optimized', '')
                
            with st.spinner('正在分析特征重要性...'):
                analyzer = SHAPAnalyzer(save_dir="results/shap")
                model_type = model_info.get('model_type', 'tree')  # 默认使用树模型类型
                 
                try:
                    analyzer.explain(
                        model, 
                        X_val, 
                        feature_names=list(X_train.columns), 
                        model_type=model_type, 
                        plot_types=("bar", "summary")
                    )
                       
                    st.success(f"分析完成！模型: {selected_model_key}, 准确率: {model_info['accuracy']:.4f}")
                      
                    # 显示分析结果
                    if os.path.exists("results/shap/shap_bar.png"):
                        st.image("results/shap/shap_bar.png", caption="特征重要性条形图")
                    if os.path.exists("results/shap/shap_beeswarm.png"):
                        st.image("results/shap/shap_beeswarm.png", caption="SHAP beeswarm图")
                    if os.path.exists("results/shap/shap_summary.png"):
                        st.image("results/shap/shap_summary.png", caption="SHAP summary图")
                           
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
                    st.info("某些模型类型可能需要更多的数据或不同的分析方法")

if __name__ == "__main__":
    run_streamlit()
