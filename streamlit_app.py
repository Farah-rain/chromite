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
st.title("🔮 铬铁矿地外来源判别系统")

# 载入模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# SHAP explainer 只需要加载一次
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

st.markdown("请上传一个与训练数据格式一致的 Excel 文件（含全部特征列）👇")

uploaded_file = st.file_uploader("📤 上传 Excel 文件", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
    st.dataframe(df.head())

    # 预测准备
    X_new = df.select_dtypes(include=[np.number])
    result_df = df.copy()

    st.markdown("---")
    st.subheader("🎯 模型预测结果")

    # 一级分类
    proba1 = model_lvl1.predict_proba(X_new)
    pred1 = model_lvl1.predict(X_new)
    conf1 = np.max(proba1, axis=1)

    result_df["Level1_Pred"] = pred1
    result_df["Level1_Prob"] = conf1

    # 二级分类（extraterrestrial 才进行）
    mask2 = pred1 == "extraterrestrial"
    X_lvl2 = X_new[mask2]

    if not X_lvl2.empty:
        proba2 = model_lvl2.predict_proba(X_lvl2)
        pred2 = model_lvl2.predict(X_lvl2)
        conf2 = np.max(proba2, axis=1)

        result_df.loc[mask2, "Level2_Pred"] = pred2
        result_df.loc[mask2, "Level2_Prob"] = conf2

        # 三级分类（OC / CC）
        mask3 = mask2 & result_df["Level2_Pred"].isin(["OC", "CC"])
        X_lvl3 = X_new[mask3]

        if not X_lvl3.empty:
            proba3 = model_lvl3.predict_proba(X_lvl3)
            pred3 = model_lvl3.predict(X_lvl3)
            conf3 = np.max(proba3, axis=1)

            result_df.loc[mask3, "Level3_Pred"] = pred3
            result_df.loc[mask3, "Level3_Prob"] = conf3

    st.dataframe(result_df[["Level1_Pred", "Level1_Prob", 
                            "Level2_Pred", "Level2_Prob", 
                            "Level3_Pred", "Level3_Prob"]].fillna("-"))

    st.markdown("---")
    st.subheader("🔍 选择一行查看 SHAP 解读")

    index = st.number_input("输入样本行号（从0开始）", min_value=0, max_value=len(result_df)-1, step=1)

    if st.button("🎨 显示 SHAP 图"):
        sample = X_new.iloc[[index]]

        fig, ax = plt.subplots(figsize=(10, 4))
        shap_values = explainer1.shap_values(sample)
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📦 样本入库（增加到训练池）")

    if st.checkbox("✅ 我确认以上预测可信，可以加入训练池"):
        save_df = result_df.copy()
        save_df["User_Confirmed"] = True

        append_path = "training_data_append.xlsx"
        if os.path.exists(append_path):
            existing = pd.read_excel(append_path)
            combined = pd.concat([existing, save_df], ignore_index=True)
        else:
            combined = save_df

        combined.to_excel(append_path, index=False)
        st.success("🎉 样本已成功保存到训练池 Excel 文件中！")

    import io

# 👇 在你下载按钮之前加这个
output = io.BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False)
st.download_button("📥 下载完整预测结果", data=output.getvalue(), file_name="predicted_results.xlsx")


