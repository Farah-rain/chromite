import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import io

# 设置页面标题
st.set_page_config(page_title="镁铁矿分类器", layout="wide")
st.title("✨ 镁铁矿 地外来源判别系统")

# 加载模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# SHAP explainer，只需要加载一次
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

# 说明上传文件格式
st.markdown("请上传一个与训练数据格式一致的 Excel 文件（含全部特征列）👇")

uploaded_file = st.file_uploader("📤 上传 Excel 文件", type=["xlsx"])
if uploaded_file is not None:
    # 读取数据
    df = pd.read_excel(uploaded_file)

    # 显示上传的数据预览
    st.subheader("📊 上传数据预览")
    st.write(df.head())

    # 模型训练用的特征列（你训练时使用的特征顺序）
    feature_cols = df.columns.tolist()

    # 一级分类预测
    pred1 = model_lvl1.predict(df[feature_cols])
    prob1 = model_lvl1.predict_proba(df[feature_cols])

    # 二级分类预测
    pred2 = model_lvl2.predict(df[feature_cols])
    prob2 = model_lvl2.predict_proba(df[feature_cols])

    # 三级分类预测
    pred3 = model_lvl3.predict(df[feature_cols])
    prob3 = model_lvl3.predict_proba(df[feature_cols])

    # 添加结果列到原始数据
    df["一级分类预测"] = pred1
    df["一级预测概率"] = np.max(prob1, axis=1)
    df["二级分类预测"] = pred2
    df["二级预测概率"] = np.max(prob2, axis=1)
    df["三级分类预测"] = pred3
    df["三级预测概率"] = np.max(prob3, axis=1)

    st.subheader("🧾 预测结果")
    st.write(df)

    # 下载链接
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载预测结果 CSV", data=csv, file_name="prediction_results.csv", mime='text/csv')

    # SHAP 图绘图函数
    def plot_shap_summary(explainer, X, title):
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(5, 5))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"{title} SHAP")
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("📈 可解释性分析（SHAP）")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 一级分类 SHAP")
        plot_shap_summary(explainer1, df[feature_cols], "一级")

    with col2:
        st.markdown("#### 二级分类 SHAP")
        plot_shap_summary(explainer2, df[feature_cols], "二级")

    with col3:
        st.markdown("#### 三级分类 SHAP")
        plot_shap_summary(explainer3, df[feature_cols], "三级")

    # 添加确认是否加入训练池
    st.subheader("✅ 是否将该样本加入训练池")
    confirm = st.checkbox("我确认这些数据可以用于再训练")

    if confirm:
        save_path = "training_pool.xlsx"
        # 如果已有文件，先读取后追加；否则直接保存
        if os.path.exists(save_path):
            existing = pd.read_excel(save_path)
            new_data = pd.concat([existing, df], ignore_index=True)
        else:
            new_data = df
        new_data.to_excel(save_path, index=False)
        st.success("🎉 数据已成功加入训练池！")
