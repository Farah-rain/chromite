import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import io
import os

st.set_page_config(page_title="镁铁矿分类系统", layout="wide")
st.title("✨ 镁铁矿地外来源判别系统")

# 加载模型
model_lvl1 = load("model_level1.pkl")
model_lvl2 = load("model_level2.pkl")
model_lvl3 = load("model_level3.pkl")

# 加载原始类别标签（你需要确保这些映射与你模型训练时一致）
labels_lvl2 = ['CV', 'CO', 'CK', 'CR', 'CM', 'CI', 'CH', 'CB', 'Other']
labels_lvl3 = ['CV_red', 'CV_ox', 'CO3.0', 'CO3.3', 'CK_A', 'CK_B', 'CM1', 'CM2', 'CB3']

# 创建explainer
explainer1 = shap.TreeExplainer(model_lvl1)
explainer2 = shap.TreeExplainer(model_lvl2)
explainer3 = shap.TreeExplainer(model_lvl3)

# 读取特征列
with open("feature_list.txt", "r", encoding="utf-8") as f:
    feature_cols = f.read().splitlines()

st.markdown("请上传一个与训练特征兼容的 CSV 文件👇")
uploaded_file = st.file_uploader("", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        df = df.fillna(0)  # 缺失值填充为0
        df_features = df[feature_cols] if all(col in df.columns for col in feature_cols) else df

        pred_lvl1 = model_lvl1.predict(df_features)
        pred_lvl2 = model_lvl2.predict(df_features)
        pred_lvl3 = model_lvl3.predict(df_features)

        prob_lvl1 = model_lvl1.predict_proba(df_features)
        prob_lvl2 = model_lvl2.predict_proba(df_features)
        prob_lvl3 = model_lvl3.predict_proba(df_features)

        # 展示预测结果
        st.subheader("🎯 预测结果：")
        result_df = df.copy()
        result_df["一级分类"] = pred_lvl1
        result_df["二级分类"] = [labels_lvl2[i] for i in pred_lvl2]
        result_df["三级分类"] = [labels_lvl3[i] for i in pred_lvl3]
        st.dataframe(result_df)

        # SHAP 绘图函数（条形图）
        def plot_shap_bar(explainer, X, title, labels):
            shap_values = explainer.shap_values(X)
            fig, ax = plt.subplots(figsize=(5, 5))
            if isinstance(shap_values, list):
                for i, class_vals in enumerate(shap_values):
                    vals = np.abs(class_vals).mean(0)
                    ax.barh(range(len(X.columns)), vals, label=labels[i], left=np.sum([np.abs(shap_values[j]).mean(0) for j in range(i)], axis=0) if i > 0 else 0)
                ax.set_yticks(range(len(X.columns)))
                ax.set_yticklabels(X.columns)
                ax.invert_yaxis()
                ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
                ax.set_title(title)
                ax.legend()
            else:
                vals = np.abs(shap_values).mean(0)
                ax.barh(range(len(X.columns)), vals)
                ax.set_yticks(range(len(X.columns)))
                ax.set_yticklabels(X.columns)
                ax.invert_yaxis()
                ax.set_xlabel("mean(|SHAP value|)")
                ax.set_title(title)
            st.pyplot(fig)

        st.subheader("📊 可解释性分析（SHAP）")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("一级分类 SHAP")
            plot_shap_bar(explainer1, df_features, "SHAP - Level1", labels=None)
        with col2:
            st.markdown("二级分类 SHAP")
            plot_shap_bar(explainer2, df_features, "SHAP - Level2", labels_lvl2)
        with col3:
            st.markdown("三级分类 SHAP")
            plot_shap_bar(explainer3, df_features, "SHAP - Level3", labels_lvl3)

        # 确认加入训练池
        st.subheader("🤝 加入训练池")
        if st.button("确认将本次样本加入训练池"):
            with open("new_training_pool.csv", "a", encoding="utf-8", newline="") as f:
                result_df.to_csv(f, index=False, header=f.tell() == 0)
            st.success("✅ 已成功加入训练池！")

    except Exception as e:
        st.error(f"发生错误：{str(e)}")

else:
    st.warning("请先上传数据文件。")
