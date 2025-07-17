import streamlit as st
import pandas as pd
import joblib
import shap
import io

# 加载模型
model_lvl1 = joblib.load("model_level1.pkl")
model_lvl2 = joblib.load("model_level2.pkl")
model_lvl3 = joblib.load("model_level3.pkl")

# 页面设置
st.set_page_config(page_title="铬铁矿 地外来源判别系统", layout="wide")
st.title("✨ 铬铁矿 地外来源判别系统")

# 上传文件
uploaded_file = st.file_uploader("📤 请上传需要预测的 CSV 文件（包含所有特征列）", type=["xlsx", "xls"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("📥 已上传数据预览：")
    st.write(input_df.head())

    # 一级模型预测
    pred_lvl1 = model_lvl1.predict(input_df)
    input_df["Level 1"] = pred_lvl1

    # 二级模型预测（只对 Level 1 = 'extraterrestrial' 的数据）
    mask_lvl2 = input_df["Level 1"] == "extraterrestrial"
    input_df_lvl2 = input_df[mask_lvl2].copy()
    if not input_df_lvl2.empty:
        pred_lvl2 = model_lvl2.predict(input_df_lvl2)
        input_df.loc[mask_lvl2, "Level 2"] = pred_lvl2

    # 三级模型预测（只对 Level 2 = 'OC' 或 'CC' 的数据）
    mask_lvl3 = input_df["Level 2"].isin(["OC", "CC"])
    input_df_lvl3 = input_df[mask_lvl3].copy()
    if not input_df_lvl3.empty:
        pred_lvl3 = model_lvl3.predict(input_df_lvl3)
        input_df.loc[mask_lvl3, "Level 3"] = pred_lvl3

    result_df = input_df

    st.success("🎉 预测完成！")
    st.write("🧾 完整预测结果预览：")
    st.write(result_df)

    # 判断是否为空，避免空数据写入 Excel 报错
    if result_df.empty:
        st.error("❌ 当前没有预测结果，无法导出 Excel 文件。请检查上传数据是否正确。")
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False)

        st.download_button(
            "⬇️ 下载完整预测结果",
            data=output.getvalue(),
            file_name="predicted_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
