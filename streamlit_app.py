import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import io

# 设置页面标题和布局
st.set_page_config(page_title="铬铁矿分类器", layout="wide")
st.title("✨ 铬铁矿地外来源判别系统")

# ======== 载入模型 ============
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# SHAP explainer 只需要加载一次
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

# ========== 文件上传 ==========
st.markdown("请上传一个与训练集数据格式一致的 Excel 文件（含全部特征列）👇")

uploaded_file = st.file_uploader("📁 上传 Excel 文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("仅支持 .csv 或 .xlsx 文件，请重新上传")
            st.stop()
        st.success(f"✅ 成功读取数据，共 {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")
        st.stop()
    
    # ========== 模型预测 ==========
    st.subheader("🔍 模型预测结果")
    X = df.copy()
    pred1 = model_lvl1.predict(X)
    pred2 = model_lvl2.predict(X)
    pred3 = model_lvl3.predict(X)
    prob3 = model_lvl3.predict_proba(X)

    df["一级分类"] = pred1
    df["二级分类"] = pred2
    df["三级分类"] = pred3
    df["地外概率"] = np.max(prob3, axis=1)

    st.dataframe(df)

    # ========== SHAP可视化 ==========
    st.subheader("📊 可解释性分析（SHAP）")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("一级分类 SHAP")
        shap_vals1 = explainer1.shap_values(X)
        fig1 = plt.figure()
        shap.summary_plot(shap_vals1, X, show=False)
        st.pyplot(fig1)

    with col2:
        st.markdown("三级分类 SHAP")
        shap_vals3 = explainer3.shap_values(X)
        fig3 = plt.figure()
        shap.summary_plot(shap_vals3, X, show=False)
        st.pyplot(fig3)

    # ========== 导出预测结果 ==========
    st.subheader("📥 下载预测结果")
    to_download = df.copy()
    to_download.to_excel("prediction_results.xlsx", index=False)
    with open("prediction_results.xlsx", "rb") as f:
        st.download_button("📩 下载 Excel 结果", f, file_name="chromite_prediction.xlsx")

else:
    st.info("请先上传数据文件以开始预测。")
