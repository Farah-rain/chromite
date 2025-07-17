import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import io

# 页面设置
st.set_page_config(page_title="铬铁矿分类器", layout="wide")
st.title("✨ 铬铁矿 地外来源判别系统")

# 加载模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# 初始化 SHAP explainer
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

# 文件上传
st.markdown("请上传一个与训练集数据格式一致的 Excel 或 CSV 文件（含全部特征列）👇")
uploaded_file = st.file_uploader("📁 上传 Excel/CSV 文件", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")
        st.stop()

    # 模型预测
    pred1 = model_lvl1.predict(df)
    prob1 = model_lvl1.predict_proba(df)

    pred2 = model_lvl2.predict(df)
    prob2 = model_lvl2.predict_proba(df)

    pred3 = model_lvl3.predict(df)
    prob3 = model_lvl3.predict_proba(df)

    # 展示预测结果
    df["一级分类"] = pred1
    df["一级分类概率"] = prob1.max(axis=1)
    df["二级分类"] = pred2
    df["二级分类概率"] = prob2.max(axis=1)
    df["三级分类"] = pred3
    df["三级分类概率"] = prob3.max(axis=1)

    st.subheader("🔍 预测结果")
    st.dataframe(df[["一级分类", "一级分类概率", "二级分类", "二级分类概率", "三级分类", "三级分类概率"] + df.columns.tolist()[:-6]])

    # SHAP 可解释性图（每类一张图）
    def plot_shap_summary(explainer, X, title):
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.summary_plot(shap_values, X, plot_type="dot", show=False)
        st.pyplot(fig)

    st.subheader("📊 可解释性分析（SHAP）")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 一级分类 SHAP")
        plot_shap_summary(explainer1, df, "一级")

    with col2:
        st.markdown("#### 二级分类 SHAP")
        plot_shap_summary(explainer2, df, "二级")

    with col3:
        st.markdown("#### 三级分类 SHAP")
        plot_shap_summary(explainer3, df, "三级")

    # 是否加入训练池
    st.subheader("📥 是否将本批数据加入训练池？")
    if st.button("✅ 确认加入"):
        if not os.path.exists("training_pool.csv"):
            df.to_csv("training_pool.csv", index=False)
        else:
            df_old = pd.read_csv("training_pool.csv")
            df_all = pd.concat([df_old, df], ignore_index=True)
            df_all.to_csv("training_pool.csv", index=False)
        st.success("已加入训练池！✨")

    elif st.button("❌ 不加入"):
        st.info("你选择了不加入训练池。")

    # 下载结果
    st.subheader("📥 下载预测结果")
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        label="📤 下载 Excel 结果",
        data=output.getvalue(),
        file_name="prediction_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
