import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib
import requests
import base64
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

st.set_page_config(page_title="铬铁矿地外来源判别系统", layout="wide")
st.title("✨ 铬铁矿 地外来源判别系统")

# ⛏️ 模型和特征加载
@st.cache_resource
def load_model_and_metadata():
    model_lvl1 = joblib.load("model_level1.pkl")
    model_lvl2 = joblib.load("model_level2.pkl")
    model_lvl3 = joblib.load("model_level3.pkl")

    features = model_lvl1.feature_name_  # 所有模型共用相同特征列

    le1 = LabelEncoder().fit(model_lvl1.classes_)
    le2 = LabelEncoder().fit(model_lvl2.classes_)
    le3 = LabelEncoder().fit(model_lvl3.classes_)

    return model_lvl1, model_lvl2, model_lvl3, features, le1, le2, le3

model_lvl1, model_lvl2, model_lvl3, feature_list, le1, le2, le3 = load_model_and_metadata()

# 📤 上传数据
uploaded_file = st.file_uploader("请上传待预测的 Excel 或 CSV 文件（包含所有特征列）", type=["xlsx", "csv"])

# 🔍 预测函数
def predict_all_levels(df):
    df_input = df.copy()
    for col in feature_list:
        if col not in df_input.columns:
            df_input[col] = np.nan
    df_input = df_input[feature_list].astype(float)

    # 一级分类
    prob1 = model_lvl1.predict_proba(df_input)
    pred1_idx = np.argmax(prob1, axis=1)
    pred1_label = le1.inverse_transform(pred1_idx)

    # 二级分类（仅限一级为 extraterrestrial）
    mask_lvl2 = (pred1_label == "extraterrestrial")
    df_lvl2 = df_input[mask_lvl2]
    prob2 = np.full((len(df_input), len(le2.classes_)), np.nan)
    pred2_label = np.full(len(df_input), "", dtype=object)
    if len(df_lvl2) > 0:
        prob2_masked = model_lvl2.predict_proba(df_lvl2)
        idx2 = np.argmax(prob2_masked, axis=1)
        pred2_masked = le2.inverse_transform(idx2)
        prob2[mask_lvl2] = prob2_masked
        pred2_label[mask_lvl2] = pred2_masked

    # 三级分类（仅限二级为 OC 或 CC）
    mask_lvl3 = (pred2_label == "OC") | (pred2_label == "CC")
    df_lvl3 = df_input[mask_lvl3]
    prob3 = np.full((len(df_input), len(le3.classes_)), np.nan)
    pred3_label = np.full(len(df_input), "", dtype=object)
    if len(df_lvl3) > 0:
        prob3_masked = model_lvl3.predict_proba(df_lvl3)
        idx3 = np.argmax(prob3_masked, axis=1)
        pred3_masked = le3.inverse_transform(idx3)
        prob3[mask_lvl3] = prob3_masked
        pred3_label[mask_lvl3] = pred3_masked

    # 📊 结果展示
    result = df.copy()
    result.insert(0, "Level1_预测", pred1_label)
    result.insert(1, "Level2_预测", pred2_label)
    result.insert(2, "Level3_预测", pred3_label)

    for i, c in enumerate(le1.classes_):
        result[f"P_Level1_{c}"] = prob1[:, i]
    for i, c in enumerate(le2.classes_):
        result[f"P_Level2_{c}"] = prob2[:, i]
    for i, c in enumerate(le3.classes_):
        result[f"P_Level3_{c}"] = prob3[:, i]

    st.subheader("🧾 预测结果：")
    st.dataframe(result)

    # 📈 可解释性分析（SHAP）
    st.subheader("📈 可解释性分析（SHAP）")
    cols = st.columns(3)
    for i, (model, name, le) in enumerate(zip([model_lvl1, model_lvl2, model_lvl3], ["Level1", "Level2", "Level3"], [le1, le2, le3])):
        with cols[i]:
            st.markdown(f"#### 🔍 {name} 模型 SHAP 解释")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)
            class_names = le.inverse_transform(np.arange(len(le.classes_)))

            fig = plt.figure(figsize=(4, 3))
            shap.summary_plot(shap_values, df_input, plot_type="bar", class_names=class_names, show=False)
            st.pyplot(fig)
            plt.clf()

    # ✅ 确认加入训练池
    st.subheader("🧩 是否将预测样本加入训练池？")
    if st.checkbox("✅ 确认将这些样本加入训练池用于再训练"):
        df_save = df_input.copy()
        df_save["Level1"] = pred1_label
        df_save["Level2"] = pred2_label
        df_save["Level3"] = pred3_label
        df_save.to_csv("training_pool.csv", mode="a", header=not os.path.exists("training_pool.csv"), index=False, encoding="utf-8-sig")
        st.success("✅ 样本已加入训练池！")

        # 📤 自动上传至 GitHub
        try:
            GITHUB_TOKEN = st.secrets["github"]["token"]
            repo_owner = "Farah-rain"
            repo_name = "chromite"
            file_path = "training_pool.csv"
            commit_msg = "update training pool"

            with open(file_path, "rb") as f:
                content = f.read()
                content_b64 = base64.b64encode(content).decode("utf-8")

            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json"
            }

            r = requests.get(url, headers=headers)
            sha = r.json()["sha"] if r.status_code == 200 else None

            data = {
                "message": commit_msg,
                "content": content_b64,
                "branch": "main"
            }
            if sha:
                data["sha"] = sha

            put_resp = requests.put(url, headers=headers, json=data)
            if put_resp.status_code in [200, 201]:
                st.success("✅ 已同步上传至 GitHub 仓库！")
            else:
                st.warning(f"⚠️ GitHub 上传失败：{put_resp.json()}")
        except Exception as e:
            st.error(f"❌ GitHub 上传失败：{e}")

    # 📥 提供下载预测结果 Excel（按钮在最后）
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        result.to_excel(writer, index=False, sheet_name='Prediction')
    st.download_button(
        label="📥 下载预测结果 Excel",
        data=output.getvalue(),
        file_name="prediction_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 🔄 主逻辑
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)
        predict_all_levels(df_uploaded)
    except Exception as e:
        st.error(f"❌ 错误：{str(e)}")
