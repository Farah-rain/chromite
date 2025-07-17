import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="铬铁矿地外来源判别系统", layout="wide")
st.title("✨ 铬铁矿 地外来源判别系统")

# 文件上传
uploaded_file = st.file_uploader("📤 请上传需要预测的 CSV 文件（包含所有特征列）", type=["csv", "xlsx"])

# 加载模型和特征列
@st.cache_resource
def load_model_and_features():
    model = lgb.Booster(model_file="best_model.txt")
    with open("feature_list.txt", "r", encoding="utf-8") as f:
        feature_list = f.read().splitlines()
    return model, feature_list

model, feature_list = load_model_and_features()

# 预测函数
def predict_and_plot(df, level_name):
    # 自动补齐或删除列
    df = df.copy()
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0  # 缺的列补零
    df = df[feature_list]  # 丢弃多余列

    # 预测概率和类别
    y_pred_prob = model.predict(df)
    y_pred_label = y_pred_prob.argmax(axis=1)

    st.subheader("🌟 分类预测结果")
    st.dataframe(pd.DataFrame({"预测类别": y_pred_label, "预测概率": y_pred_prob.max(axis=1)}))

    # 是否加入训练池
    if st.checkbox("✅ 确认将这些样本加入训练池，用于未来再训练？"):
        df_insert = df.copy()
        df_insert[level_name] = y_pred_label
        df_insert.to_csv("training_pool.csv", mode="a", index=False, header=not os.path.exists("training_pool.csv"), encoding="utf-8-sig")
        st.success("已成功加入训练池！")

    # SHAP 分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    st.subheader("📊 可解释性分析（SHAP）")
    for i, class_shap in enumerate(shap_values):
        shap.summary_plot(class_shap, df, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

# 主逻辑
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        predict_and_plot(df, level_name="一级分类")
    except Exception as e:
        st.error(f"发生错误：{str(e)}")
