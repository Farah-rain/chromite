
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

# -------------------- 页面配置（仅 UI 英文） --------------------
st.set_page_config(page_title="Chromite Extraterrestrial Origin Classifier", layout="wide")
st.title("✨ Chromite Extraterrestrial Origin Classifier")

# -------------------- 常量与映射（与训练一致；注释中文） --------------------
ABSTAIN_LABEL = "Unknown"  # Unknown 标签统一口径
THRESHOLDS = {"Level2": 0.90, "Level3": 0.90}  # Level2/Level3 的放行阈值
# 父类键与子类集合（注意：若模型没有聚合类 "UOC"，需改成 UOC-H/L/LL）
valid_lvl3 = {
    "OC": {"EOC-H", "EOC-L", "EOC-LL", "UOC"},
    "CC": {"CM-CO", "CR-clan", "CV"}
}

# -------------------- 小工具函数（注释中文） --------------------
def apply_threshold(proba: np.ndarray, classes: np.ndarray, thr: float):
    """对分类概率应用阈值：最大概率>=thr 时输出该类别，否则 Unknown。返回(预测, 最大概率)。"""
    max_idx = np.argmax(proba, axis=1)
    max_val = proba[np.arange(proba.shape[0]), max_idx]
    pred = np.where(max_val >= thr, classes[max_idx], ABSTAIN_LABEL)
    return pred, max_val

@st.cache_resource
def load_model_and_metadata():
    """模型与特征列加载：优先 models/ 目录；特征列优先 JSON 退回 model.feature_name_。"""
    def _load(path1, path2):
        return joblib.load(path1) if os.path.exists(path1) else joblib.load(path2)

    model_lvl1 = _load("models/model_level1.pkl", "model_level1.pkl")
    model_lvl2 = _load("models/model_level2.pkl", "model_level2.pkl")
    model_lvl3 = _load("models/model_level3.pkl", "model_level3.pkl")

    feat_json = "models/feature_columns.json" if os.path.exists("models/feature_columns.json") else "feature_columns.json"
    if os.path.exists(feat_json):
        import json
        with open(feat_json, "r", encoding="utf-8") as f:
            features = json.load(f)
    else:
        features = getattr(model_lvl1, "feature_name_", None)
        if not features:
            st.error("Feature columns not found (feature_columns.json or model.feature_name_).")
            st.stop()
    return model_lvl1, model_lvl2, model_lvl3, features

# ==== 新的 explainer 缓存实现：用“签名 + 下划线参数”避免对模型对象做哈希 ====
@st.cache_resource
def _make_explainer_cached(sig: str, _model):
    """缓存 SHAP explainer；sig 作为缓存键，_model 不参与哈希。"""
    return shap.TreeExplainer(_model)

def _model_signature(model) -> str:
    """构造一个可哈希的模型签名：模型类名 + 排序后的超参 + 类别列表"""
    try:
        params = model.get_params()
        params_tup = tuple(sorted((k, str(v)) for k, v in params.items()))
    except Exception:
        params_tup = ()
    try:
        classes = tuple(map(str, getattr(model, "classes_", ())))
    except Exception:
        classes = ()
    return f"{model.__class__.__name__}|{hash(params_tup)}|{hash(classes)}"

def preprocess_uploaded_data(df):
    """
    数据预处理：兼容 FeOT 缺失的拆分，派生特征生成（与你原逻辑一致）。
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
        # 若已给出 FeO/Fe2O3，直接重命名并计算总 FeO
        df = df.rename(columns={"FeO": "FeOre", "Fe2O3": "Fe2O3re"})
        df["FeO_total"] = df["FeOre"] + df["Fe2O3re"] * 0.8998
    else:
        # 若只有 FeOT，则按尖晶石配位假设拆分 Fe2+/Fe3+
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

    # 派生特征（与你原版一致）
    df["Cr_CrplusAl"] = Cr_mol / (Cr_mol + Al_mol)
    df["Mg_MgplusFe"] = Mg_mol / (Mg_mol + Fe2_mol)
    df["FeCrAlFe"]   = Fe3_mol / (Fe3_mol + Cr_mol + Al_mol)
    df["FeMgFe"]     = Fe2_mol / (Fe2_mol + Mg_mol)
    return df

def to_numeric_df(df):
    """尽量把所有列转 float，无法转换则置 NaN。"""
    return df.apply(pd.to_numeric, errors="coerce")

# ========= 组结果：单层多数票 + 平均概率 =========
def summarize_level_top_share(labels, maxp):
    """
    labels: 该层每行的预测（如 Level3 的 pred3_label）
    maxp:   该层每行“被判定类别”的概率（阈值/约束后的那一列；未参与该层的行可为 NaN）

    返回:
      None  -> 没有有效标签（全 Unknown/NaN/空）
      dict  -> {"label","agree","total","share","prob"}
               prob = 仅在该类别的那些行上的平均概率
    """
    s = pd.Series(labels, dtype="object")
    m = s.notna() & (s != "") & (s != ABSTAIN_LABEL)
    total = int(m.sum())
    if total == 0:
        return None

    s_valid = s[m]
    p_valid = pd.Series(maxp, dtype="float64")[m]

    counts = s_valid.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()

    means = {c: float(p_valid[s_valid == c].mean()) for c in candidates}
    best = max(candidates, key=lambda c: means[c])

    return {"label": best, "agree": int(max_count), "total": total,
            "share": max_count/total, "prob": means[best]}

# -------------------- 加载模型 & 特征（侧边栏） --------------------
with st.sidebar:
    st.subheader("Model Loading")
    try:
        model_lvl1, model_lvl2, model_lvl3, feature_list = load_model_and_metadata()
        st.success("Models and feature list loaded.")
        st.caption(f"Feature dimension: {len(feature_list)}")
    except Exception as e:
        st.error("Failed to load models or feature columns.")
        st.exception(e)

# -------------------- 上传文件并处理（UI 英文） --------------------
uploaded_file = st.file_uploader("Upload an Excel or CSV file (must include all feature columns).", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)

        # 预处理（含 Fe 拆分 & 派生特征）
        df_uploaded = preprocess_uploaded_data(df_uploaded)

        # 对齐特征列（缺失补 NaN，按训练列顺序取）
        df_input = df_uploaded.copy()
        for col in feature_list:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_input = to_numeric_df(df_input[feature_list])

        # -------------------- 三级推理（与训练口径一致） --------------------
        # Level1（不启用 Unknown）
        prob1 = model_lvl1.predict_proba(df_input)
        classes1 = model_lvl1.classes_.astype(str)
        pred1_idx = np.argmax(prob1, axis=1)
        pred1_label = classes1[pred1_idx]
        p1max = prob1[np.arange(len(df_input)), pred1_idx]  # L1 最大概率

        # Level2（仅 L1=Extraterrestrial；阈值放行 -> Unknown）
        _pred1_norm = (
            pd.Series(pred1_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        )
        mask_lvl2 = (_pred1_norm == "extraterrestrial").to_numpy()

        prob2 = np.full((len(df_input), len(model_lvl2.classes_)), np.nan)
        pred2_label = np.full(len(df_input), "", dtype=object)
        p2max = np.full(len(df_input), np.nan)

        if mask_lvl2.any():
            prob2_masked = model_lvl2.predict_proba(df_input[mask_lvl2])
            classes2 = model_lvl2.classes_.astype(str)
            pred2_masked, p2max_masked = apply_threshold(prob2_masked, classes2, THRESHOLDS["Level2"])
            prob2[mask_lvl2] = prob2_masked
            pred2_label[mask_lvl2] = pred2_masked
            p2max[mask_lvl2] = p2max_masked

        # Level3（父子约束 + 阈值放行）
        _pred2_norm = (
            pd.Series(pred2_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        )
        mask_lvl3 = _pred2_norm.isin(["oc", "cc"]).to_numpy()

        prob3 = np.full((len(df_input), len(model_lvl3.classes_)), np.nan)
        pred3_label = np.full(len(df_input), "", dtype=object)
        p3max = np.full(len(df_input), np.nan)

        if mask_lvl3.any():
            all_proba3 = model_lvl3.predict_proba(df_input[mask_lvl3])
            classes3 = model_lvl3.classes_.astype(str)
            idxs = np.where(mask_lvl3)[0]
            for row_i, i_global in enumerate(idxs):
                parent = str(pred2_label[i_global])          # "OC" 或 "CC"
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
                p3max[i_global] = pmax
            prob3[mask_lvl3] = all_proba3

        # -------------------- 展示结果（列名英文） --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "Index", df_display.index + 1)
        df_display.insert(1, "Level1_Pred", pred1_label)
        df_display.insert(2, "Level2_Pred", pred2_label)
        df_display.insert(3, "Level3_Pred", pred3_label)

        for i, c in enumerate(model_lvl1.classes_.astype(str)):
            df_display[f"P_Level1_{c}"] = prob1[:, i]
        for i, c in enumerate(model_lvl2.classes_.astype(str)):
            df_display[f"P_Level2_{c}"] = prob2[:, i]
        for i, c in enumerate(model_lvl3.classes_.astype(str)):
            df_display[f"P_Level3_{c}"] = prob3[:, i]

        st.subheader("🧾 Predictions")
        st.dataframe(df_display)

        # -------------------- SHAP 可解释性 --------------------
        st.subheader("📈 SHAP Interpretability")
        cols = st.columns(3)
        for col, (model, name) in zip(cols, [(model_lvl1, "Level1"), (model_lvl2, "Level2"), (model_lvl3, "Level3")]):
            with col:
                st.markdown(f"#### 🔍 {name} Model")
                explainer = _make_explainer_cached(_model_signature(model), _model=model)
                shap_values = explainer.shap_values(df_input)
                shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
                st.pyplot(plt.gcf()); plt.close()
                shap.summary_plot(shap_values, df_input, show=False)
                st.pyplot(plt.gcf()); plt.close()

        # -------------------- ✅ 新板块：样品一致性确认 + 组结果（三级都比较） --------------------
        st.subheader("🧪 Specimen Confirmation & Group Result")
        same_specimen = st.checkbox("I confirm all uploaded rows originate from the same physical specimen.")
        if same_specimen:
            # 分别计算 Level1/2/3 的多数票+平均概率
            sum_L1 = summarize_level_top_share(pred1_label, p1max)
            sum_L2 = summarize_level_top_share(pred2_label, p2max)
            sum_L3 = summarize_level_top_share(pred3_label, p3max)

            cands = []
            if sum_L1: cands.append(("Level1", sum_L1))
            if sum_L2: cands.append(("Level2", sum_L2))
            if sum_L3: cands.append(("Level3", sum_L3))

            if cands:
                # 先比占比 share，再比平均概率 prob，最后偏向更细层级 L3>L2>L1
                depth = {"Level1": 1, "Level2": 2, "Level3": 3}
                final_level, final = sorted(
                    cands,
                    key=lambda t: (t[1]["share"], t[1]["prob"], depth[t[0]]),
                    reverse=True
                )[0]

                st.success(
                    f"Final group result → **{final_level}: {final['label']}**  |  "
                    f"Probability (mean for this class): **{final['prob']:.3f}**  |  "
                    f"Share: **{final['agree']}/{final['total']} ({final['share']:.0%})**"
                )

                # 小表：三层对比，心里更有底
                rows = []
                for lvl, s in [("Level1", sum_L1), ("Level2", sum_L2), ("Level3", sum_L3)]:
                    if s:
                        rows.append({
                            "Level": lvl,
                            "Top class": s["label"],
                            "Share": f"{s['agree']}/{s['total']} ({s['share']:.0%})",
                            "Mean prob": round(s["prob"], 3),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No valid predictions available to summarize this group.")

        # -------------------- 训练池 & GitHub 同步 --------------------
        st.subheader("🧩 Add Predictions to Training Pool?")
        if st.checkbox("✅ Confirm to append these samples to the training pool for future retraining"):
            df_save = df_input.copy()
            df_save["Level1"] = pred1_label
            df_save["Level2"] = pred2_label
            df_save["Level3"] = pred3_label
            local_path = "training_pool.csv"
            header_needed = not os.path.exists(local_path)
            df_save.to_csv(local_path, mode="a", header=header_needed, index=False, encoding="utf-8-sig")
            st.success("✅ Samples appended to local training pool.")

            try:
                GITHUB_TOKEN = (
                    st.secrets.get("gh_token")
                    or (st.secrets.get("github", {}) or {}).get("token")
                )
                repo_owner = st.secrets.get("gh_repo_owner", "Farah-rain")
                repo_name  = st.secrets.get("gh_repo_name",  "chromite")
                dst_path   = st.secrets.get("gh_dst_path",   "training_pool.csv")
                branch     = st.secrets.get("gh_branch",     "main")

                if not GITHUB_TOKEN:
                    st.info("GitHub token not configured (gh_token or github.token). Saved locally only.")
                else:
                    with open(local_path, "rb") as f:
                        content_b64 = base64.b64encode(f.read()).decode("utf-8")
                    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{dst_path}"
                    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
                    r = requests.get(url, headers=headers)
                    sha = r.json().get("sha") if r.status_code == 200 else None
                    payload = {"message": "update training pool", "content": content_b64, "branch": branch}
                    if sha: payload["sha"] = sha
                    put_resp = requests.put(url, headers=headers, json=payload)
                    if 200 <= put_resp.status_code < 300:
                        st.success("✅ Synced to GitHub repository.")
                    else:
                        st.warning(f"⚠️ GitHub sync failed ({put_resp.status_code}): {put_resp.text[:300]}")
            except Exception as e:
                st.error(f"❌ GitHub sync error: {e}")

        # -------------------- 结果下载 --------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Prediction')
        st.download_button(
            label="📥 Download Predictions (Excel)",
            data=output.getvalue(),
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error("Error while processing the uploaded file.")
        st.exception(e)
else:
    st.info("Please upload a data file to proceed.")
