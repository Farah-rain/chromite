import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import io

st.set_page_config(page_title="铬铁矿分类器", layout="wide")
st.title("✨ 铬铁矿地外来源判别系统")

# 加载模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# SHAP explainer 初始化
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

st.markdown("请上传一个与训练集数据格式一致的 Excel 或 CSV 文件（包含所有特征列）👇")

uploaded_file = st.file_uploader("📎 上传文件", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"成功读取数据，共 {df.shape[0]} 行 {df.shape[1]} 列")

        # 保存原始数据备份
        original_df = df.copy()

        # 预测
        X = df.drop(columns=["编号"], errors="ignore")
        pred_lvl1 = model_lvl1.predict(X)
        pred_lvl2 = model_lvl2.predict(X)
        pred_lvl3 = model_lvl3.predict(X)

        prob_lvl1 = model_lvl1.predict_proba(X)
        prob_lvl2 = model_lvl2.predict_proba(X)
        prob_lvl3 = model_lvl3.predict_proba(X)

        df["一级分类"] = pred_lvl1
        df["一级概率"] = prob_lvl1.max(axis=1)
        df["二级分类"] = pred_lvl2
        df["二级概率"] = prob_lvl2.max(axis=1)
        df["三级分类"] = pred_lvl3
        df["三级概率"] = prob_lvl3.max(axis=1)

        st.subheader("📋 预测结果")
        st.dataframe(df)

        # 可视化 SHAP 值
        st.subheader("📊 可解释性分析（SHAP）")

        def plot_shap_summary(explainer, X, title):
            shap_values = explainer.shap_values(X)
            fig, ax = plt.subplots(figsize=(6, 6))
            shap.summary_plot(shap_values, X, plot_type="dot", show=False)
            st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("一级分类 SHAP")
            plot_shap_summary(explainer1, X, "一级")
        with col2:
            st.markdown("二级分类 SHAP")
            plot_shap_summary(explainer2, X, "二级")
        with col3:
            st.markdown("三级分类 SHAP")
            plot_shap_summary(explainer3, X, "三级")

        # 用户反馈：是否加入训练池
        st.subheader("🧠 是否将数据加入训练池")

        selected_rows = st.multiselect("请选择要加入训练池的样本编号：", df.index.astype(str))
        if selected_rows:
            confirmed_data = df.loc[df.index.astype(str).isin(selected_rows)]
            st.write("以下数据将被加入训练池：")
            st.dataframe(confirmed_data)

            # 保存确认加入的数据（可自定义路径）
            confirmed_data.to_csv("confirmed_for_training.csv", index=False)
            st.success("已保存确认数据到 confirmed_for_training.csv")

        # 下载完整预测结果
        st.subheader("📥 下载预测结果")
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button(
            label="📤 下载 Excel 结果文件",
            data=buffer.getvalue(),
            file_name="prediction_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ 处理文件时发生错误：{e}")
