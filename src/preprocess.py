"""
OULAD数据集专用数据预处理与特征工程（多表融合，兼容官方字段结构）
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(data_dir="data/anonymisedData", n_weeks=4):
    """
    加载OULAD主要数据表，自动补全学生评估表的课程信息，融合多表特征。
    参数：
        data_dir (str): 数据文件夹路径，默认为 'data/anonymisedData'。
        n_weeks (int): 统计前n周的VLE行为特征，默认为4。
    返回：
        df (pd.DataFrame): 合并后的原始特征数据表。
    """
    # 读取各表
    student_info = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    student_registration = pd.read_csv(os.path.join(data_dir, "studentRegistration.csv"))
    student_vle = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    student_assessment = pd.read_csv(os.path.join(data_dir, "studentAssessment.csv"))
    assessments = pd.read_csv(os.path.join(data_dir, "assessments.csv"))

    # 1. 先将studentAssessment与assessments表通过id_assessment关联，补全课程信息
    sa_full = pd.merge(student_assessment, assessments[["id_assessment", "code_module", "code_presentation"]],
                        on="id_assessment", how="left")

    # 2. 统计每个学生-课程-学期的平均分和作业数
    sa_stats = sa_full.groupby(["id_student", "code_module", "code_presentation"]).agg(
        avg_score=("score", "mean"),
        num_assessments=("score", "count")
    ).reset_index()

    # 3. 统计前n_weeks的VLE活跃度
    student_vle = student_vle.copy()
    student_vle["week"] = student_vle["date"] // 7
    vle_nw = student_vle[student_vle["week"] < n_weeks]
    vle_stats = vle_nw.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique")
    ).reset_index()

    # 4. 合并注册信息
    df = pd.merge(student_info, student_registration[["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"]],
                    on=["id_student", "code_module", "code_presentation"], how="left")
    # 5. 合并成绩统计
    df = pd.merge(df, sa_stats, on=["id_student", "code_module", "code_presentation"], how="left")
    # 6. 合并VLE统计
    df = pd.merge(df, vle_stats, on=["id_student", "code_module", "code_presentation"], how="left")
    return df

def preprocess_data(df):
    """
    对合并后的OULAD数据进行清洗和特征工程，生成可直接用于建模的特征和标签。

    参数：
        df (pd.DataFrame): load_data 返回的合并原始数据表。
    返回：
        df (pd.DataFrame): 仅包含有效特征和标签的建模数据表，所有特征为数值型。
    步骤：
        1. 去除无 final_result 的样本。
        2. 生成学业困难标签（Fail/Withdrawn=1，其余=0）。
        3. 选择关键特征，去除缺失值。
        4. 对类别特征做独热编码。
    注意事项：
        - 返回的DataFrame可直接用于模型训练和评估。
        - 如需调整特征或标签定义，可修改本函数。
    """
    # 只保留有final_result的样本
    df = df.dropna(subset=["final_result"])
    # 生成标签：学业困难（Fail/Withdrawn）为1，其他为0
    df["label"] = df["final_result"].map({"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1})
    # 选择特征
    feature_cols = [
        "age_band", "gender", "imd_band", "disability", "num_of_prev_attempts", "studied_credits",
        "avg_score", "num_assessments", "total_clicks", "active_days"
    ]
    df = df.dropna(subset=feature_cols + ["label"])
    # 类别特征独热编码
    df = pd.get_dummies(df[feature_cols + ["label"]], columns=["age_band", "gender", "imd_band", "disability"])
    # 特征标准化
    feature_cols_all = [c for c in df.columns if c != 'label']
    scaler = StandardScaler()
    X = df[feature_cols_all]
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols_all, index=df.index)
    df_scaled['label'] = df['label'].values
    return df_scaled
