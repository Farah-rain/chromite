import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import io

# 设置页面
st.set_page_config(page_title="铬铁矿分类器", layout="wide")
st.title("✨ 铬铁矿地外来源判别系统")

# 载入模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# 特征列
with open("feature_list.txt", "r", encoding="utf-8") as f:
    feature_list = f.read().splitlines()

# 定义 SHAP 绘图函数（bar 图）
def plot_shap_bar(model, X, title):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()

# 上传文件
st.markdown("### 📥 请上传需要预测的 Excel 文件（包含所有特征列）👇")
uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls", "csv"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_ext == "csv":
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            st.error("❌ 不支持的文件格式，请上传 Excel 或 CSV 文件")
            st.stop()
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        st.stop()

    st.success(f"✅ 成功载入数据，共 {df.shape[0]} 行 {df.shape[1]} 列")

    # 处理缺失特征：只保留模型需要的特征，缺失的用 NaN 补
    input_data = df.copy()
    for col in feature_list:
        if col not in input_data.columns:
            input_data[col] = np.nan
    input_data = input_data[feature_list]

    # 显示缺失警告
    missing_cols = [col for col in feature_list if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ 当前数据缺失以下特征：{missing_cols}，将使用 NaN 填充进行预测。")

    # 模型预测
    pred_lvl1 = model_lvl1.predict(input_data)
    pred_lvl2 = model_lvl2.predict(input_data)
    pred_lvl3 = model_lvl3.predict(input_data)

    # 显示预测结果
    df_result = df.copy()
    df_result["一级分类"] = pred_lvl1
    df_result["二级分类"] = pred_lvl2
    df_result["三级分类"] = pred_lvl3

    st.markdown("### 📊 预测结果")
    st.dataframe(df_result)

    # SHAP 分析
    st.markdown("### 📈 可解释性分析（SHAP）")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("一级分类 SHAP")
        plot_shap_bar(model_lvl1, input_data, "SHAP - Level1")
    with col2:
        st.markdown("二级分类 SHAP")
        plot_shap_bar(model_lvl2, input_data, "SHAP - Level2")
    with col3:
        st.markdown("三级分类 SHAP")
        plot_shap_bar(model_lvl3, input_data, "SHAP - Level3")

    # 加入训练池（需用户确认）
    st.markdown("### 📌 是否将该数据加入训练池？")
    if st.button("✅ 确认加入"):
        # 保存文件（可以是 append 模式写入 CSV）
        pool_path = "training_pool.csv"
        if os.path.exists(pool_path):
            old = pd.read_csv(pool_path)
            new = pd.concat([old, df_result], ignore_index=True)
        else:
            new = df_result
        new.to_csv(pool_path, index=False)
        st.success("🎉 已成功加入训练池！")
    else:
        st.info("⏳ 等待确认后再加入训练池。")
