# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib
import requests
import base64
from io import BytesIO

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="铬铁矿地外来源判别系统", layout="wide")
st.title("✨ 铬铁矿 地外来源判别系统")

# -------------------- 常量与映射（对齐训练口径） --------------------
ABSTAIN_LABEL = "Unknown"
THRESHOLDS = {"Level2": 0.90, "Level3": 0.90}
# 注意：父类键为 "OC" 和 "CC"；CC 的子类包含 "CM-CO"
valid_lvl3 = {
    "OC": {"EOC-H", "EOC-L", "EOC-LL"},
    "CC": {"CM-CO", "CR-clan", "CV"}
}

# -------------------- 工具函数 --------------------
def apply_threshold(proba: np.ndarray, classes: np.ndarray, thr: float):
    max_idx = np.argmax(proba, axis=1)
    max_val = proba[np.arange(proba.shape[0]), max_idx]
    pred = np.where(max_val >= thr, classes[max_idx], ABSTAIN_LABEL)
    return pred, max_val

@st.cache_resource
def load_model_and_metadata():
    # 路径容错：优先 models/，否则根目录
    def _load(path1, path2):
        return joblib.load(path1) if os.path.exists(path1) else joblib.load(path2)

    model_lvl1 = _load("models/model_level1.pkl", "model_level1.pkl")
    model_lvl2 = _load("models/model_level2.pkl", "model_level2.pkl")
    model_lvl3 = _load("models/model_level3.pkl", "model_level3.pkl")

    # 特征列：优先 JSON，其次 model.feature_name_
    feat_json = "models/feature_columns.json" if os.path.exists("models/feature_columns.json") else "feature_columns.json"
    if os.path.exists(feat_json):
        import json
        with open(feat_json, "r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        features = getattr(model_lvl1, "feature_name_", None)
        if not features:
            st.error("未找到特征列（feature_columns.json 或 model.feature_name_），无法对齐输入数据。")
            st.stop()
    return model_lvl1, model_lvl2, model_lvl3, features

@st.cache_resource
def _make_explainer(model):
    return shap.TreeExplainer(model)

def preprocess_uploaded_data(df):
    """
    完整保留你的逻辑，并在 FeOT 缺失时安全兜底；自动生成派生特征。
    """
    MW = {'TiO2':79.866,'Al2O3':101.961,'Cr2O3':151.99,'FeO':71.844,'MnO':70.937,'MgO':40.304,'ZnO':81.38,'SiO2':60.0843,'V2O3':149.88}
    O_num = {'TiO2':2,'Al2O3':3,'Cr2O3':3,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':2,'V2O3':3}
    Cat_num={'TiO2':1,'Al2O3':2,'Cr2O3':2,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':1,'V2O3':2}
    FE2O3_OVER_FEO_FE_EQ = 159.688 / (2 * 71.844)

    for ox in MW:
        if ox not in df.columns:
            df[ox] = 0.0

    df = df.copy()

    use_manual_fe = "FeO" in df.columns and "Fe2O3" in df.columns
    if use_manual_fe:
        df = df.rename(columns={"FeO": "FeOre", "Fe2O3": "Fe2O3re"})
        df["FeO_total"] = df["FeOre"] + df["Fe2O3re"] * 0.8998
    else:
        def fe_split_spinel(row, O_basis=32):
            val_feot = row.get('FeOT', np.nan)
            val_feot = 0.0 if pd.isna(val_feot) else float(val_feot)

            moles = {ox: row[ox]/MW[ox] for ox in MW if ox != 'FeO'}
            moles['FeO'] = val_feot / MW['FeO']

            O_total = sum(moles[ox] * O_num[ox] for ox in moles)
            fac = O_basis / O_total if O_total > 0 else 0.0

            cations = {ox: moles[ox] * Cat_num[ox] * fac for ox in moles}
            S = sum(cations.values())
            T = 24.0

            Fe_total_apfu = cations['FeO']
            Fe3_apfu = max(0.0, 2 * O_basis * (1 - T / S)) if S > 0 else 0.0
            Fe3_apfu = min(Fe3_apfu, Fe_total_apfu)
            Fe2_apfu = Fe_total_apfu - Fe3_apfu

            Fe2_frac = Fe2_apfu / Fe_total_apfu if Fe_total_apfu > 0 else 0.0
            Fe3_frac = Fe3_apfu / Fe_total_apfu if Fe_total_apfu > 0 else 0.0

            FeO_wt   = Fe2_frac * val_feot
            Fe2O3_wt = Fe3_frac * val_feot * FE2O3_OVER_FEO_FE_EQ

            return pd.Series({
                'FeOre': FeO_wt,
                'Fe2O3re': Fe2O3_wt,
                'Fe2_frac': Fe2_frac,
                'Fe3_frac': Fe3_frac,
                'FeO_total': FeO_wt + Fe2O3_wt * 0.8998
            })

        df = df.join(df.apply(fe_split_spinel, axis=1))

    mol_wt = {'Cr2O3':151.99,'Al2O3':101.961,'MgO':40.304,'FeO':71.844,'Fe2O3':159.688}
    Cr_mol = df["Cr2O3"] / mol_wt["Cr2O3"] * 2
    Al_mol = df["Al2O3"] / mol_wt["Al2O3"] * 2
    Mg_mol = df["MgO"] / mol_wt["MgO"]
    Fe2_mol = df["FeOre"] / mol_wt["FeO"]
    Fe3_mol = df["Fe2O3re"] / mol_wt["Fe2O3"] * 2

    df["Cr_CrplusAl"] = Cr_mol / (Cr_mol + Al_mol)
    df["Mg_MgplusFe"] = Mg_mol / (Mg_mol + Fe2_mol)
    df["FeCrAlFe"]   = Fe3_mol / (Fe3_mol + Cr_mol + Al_mol)
    df["FeMgFe"]     = Fe2_mol / (Fe2_mol + Mg_mol)
    return df

def to_numeric_df(df):
    # 尽量把所有列转为 float，无法转换的置为 NaN
    return df.apply(pd.to_numeric, errors="coerce")

def save_training_pool(df_pred):
    path = "training_pool.csv"
    header_needed = not os.path.exists(path)
    df_pred.to_csv(path, mode="a", header=header_needed, index=False, encoding="utf-8-sig")
    return path

def push_to_github_local_file(path, repo_owner, repo_name, token, dst_path="training_pool.csv", branch="main", message="update training pool"):
    with open(path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{dst_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    # 先查是否已有文件，拿 sha
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {"message": message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    put_resp = requests.put(url, headers=headers, json=payload)
    return put_resp.status_code, put_resp.text

# -------------------- 加载模型 & 特征 --------------------
with st.sidebar:
    st.subheader("模型加载")
    try:
        model_lvl1, model_lvl2, model_lvl3, feature_list = load_model_and_metadata()
        st.success("模型与特征列已加载")
        st.caption(f"特征维度：{len(feature_list)}")
    except Exception as e:
        st.error("加载模型失败，请检查 models/ 下文件与特征列 JSON")
        st.exception(e)

# -------------------- 上传文件并处理 --------------------
uploaded_file = st.file_uploader("请上传待预测的 Excel 或 CSV 文件（包含所有特征列）", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)

        # 预处理（含 Fe 拆分 & 派生特征）
        df_uploaded = preprocess_uploaded_data(df_uploaded)

        # 对齐特征列
        df_input = df_uploaded.copy()
        for col in feature_list:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_input = df_input[feature_list]
        df_input = to_numeric_df(df_input)

        # -------------------- 三级推理（对齐训练口径） --------------------
        # Level1（不启用 Unknown）
        prob1 = model_lvl1.predict_proba(df_input)
        classes1 = model_lvl1.classes_.astype(str)
        pred1_idx = np.argmax(prob1, axis=1)
        pred1_label = classes1[pred1_idx]

        # Level2（仅 L1=Extraterrestrial；阈值放行 -> Unknown）
        mask_lvl2 = np.char.lower(np.char.strip(pred1_label)) == "extraterrestrial"
        prob2 = np.full((len(df_input), len(model_lvl2.classes_)), np.nan)
        pred2_label = np.full(len(df_input), "", dtype=object)

        if mask_lvl2.any():
            prob2_masked = model_lvl2.predict_proba(df_input[mask_lvl2])
            classes2 = model_lvl2.classes_.astype(str)
            pred2_masked, _ = apply_threshold(prob2_masked, classes2, THRESHOLDS["Level2"])
            prob2[mask_lvl2] = prob2_masked
            pred2_label[mask_lvl2] = pred2_masked

        # Level3（由预测到的二级路由 + 父子约束 + 阈值放行）
        mask_lvl3 = np.isin(np.char.lower(np.char.strip(pred2_label)), ["oc", "cc"])
        prob3 = np.full((len(df_input), len(model_lvl3.classes_)), np.nan)
        pred3_label = np.full(len(df_input), "", dtype=object)

        if mask_lvl3.any():
            all_proba3 = model_lvl3.predict_proba(df_input[mask_lvl3])
            classes3 = model_lvl3.classes_.astype(str)
            idxs = np.where(mask_lvl3)[0]
            for row_i, i_global in enumerate(idxs):
                parent = str(pred2_label[i_global])
                allowed = valid_lvl3.get(parent, set())
                p = all_proba3[row_i].copy()
                if allowed:
                    mask_allowed = np.isin(classes3, list(allowed))
                    p = p * mask_allowed
                    if p.sum() > 0:
                        p = p / p.sum()
                j = int(np.argmax(p))
                pmax = float(p[j])
                pred3_label[i_global] = classes3[j] if pmax >= THRESHOLDS["Level3"] else ABSTAIN_LABEL
            prob3[mask_lvl3] = all_proba3

        # -------------------- 展示结果 --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "序号", df_display.index + 1)
        df_display.insert(1, "Level1_预测", pred1_label)
        df_display.insert(2, "Level2_预测", pred2_label)
        df_display.insert(3, "Level3_预测", pred3_label)

        for i, c in enumerate(model_lvl1.classes_.astype(str)):
            df_display[f"P_Level1_{c}"] = prob1[:, i]
        for i, c in enumerate(model_lvl2.classes_.astype(str)):
            df_display[f"P_Level2_{c}"] = prob2[:, i]
        for i, c in enumerate(model_lvl3.classes_.astype(str)):
            df_display[f"P_Level3_{c}"] = prob3[:, i]

        st.subheader("🧾 预测结果")
        st.dataframe(df_display)

        # -------------------- SHAP 可解释性分析（稳定绘图） --------------------
        st.subheader("📈 SHAP 可解释性分析")
        cols = st.columns(3)
        for col, (model, name) in zip(cols, [(model_lvl1, "Level1"), (model_lvl2, "Level2"), (model_lvl3, "Level3")]):
            with col:
                st.markdown(f"#### 🔍 {name} 模型 SHAP 解释")
                explainer = _make_explainer(model)
                shap_values = explainer.shap_values(df_input)

                # 条形图
                shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
                st.pyplot(plt.gcf()); plt.close()

                # 点云图（可选）
                shap.summary_plot(shap_values, df_input, show=False)
                st.pyplot(plt.gcf()); plt.close()

        # -------------------- 训练池追加 & GitHub 同步 --------------------
        st.subheader("🧩 是否将预测样本加入训练池？")
        if st.checkbox("✅ 确认将这些样本加入训练池用于再训练"):
            df_save = df_input.copy()
            df_save["Level1"] = pred1_label
            df_save["Level2"] = pred2_label
            df_save["Level3"] = pred3_label
            local_path = save_training_pool(df_save)
            st.success("✅ 样本已加入本地训练池！")

            try:
                # 兼容两种 secrets 写法：gh_token 或 github.token
                GITHUB_TOKEN = (
                    st.secrets.get("gh_token")
                    or (st.secrets.get("github", {}) or {}).get("token")
                )
                repo_owner = st.secrets.get("gh_repo_owner", "Farah-rain")
                repo_name  = st.secrets.get("gh_repo_name",  "chromite")
                dst_path   = st.secrets.get("gh_dst_path",   "training_pool.csv")
                branch     = st.secrets.get("gh_branch",     "main")

                if not GITHUB_TOKEN:
                    st.info("未配置 gh_token 或 github.token，已在本地保存。")
                else:
                    status, resp = push_to_github_local_file(local_path, repo_owner, repo_name, GITHUB_TOKEN, dst_path, branch)
                    if 200 <= status < 300:
                        st.success("✅ 已同步上传至 GitHub 仓库！")
                    else:
                        st.warning(f"⚠️ GitHub 上传失败（{status}）：{resp[:300]}")

            except Exception as e:
                st.error(f"❌ GitHub 上传失败：{e}")

        # -------------------- 结果下载 --------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Prediction')
        st.download_button(
            label="📥 下载预测结果 Excel",
            data=output.getvalue(),
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error("处理文件时出错")
        st.exception(e)
else:
    st.info("请先上传数据文件。")
