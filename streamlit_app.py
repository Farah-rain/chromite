
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os, joblib, requests, base64
from io import BytesIO

# -------------------- 页面配置（仅 UI 英文） --------------------
st.set_page_config(page_title="Chromite Extraterrestrial Origin Classifier", layout="wide")
st.title("✨ Chromite Extraterrestrial Origin Classifier")

# -------------------- 常量与映射（与训练一致；注释中文） --------------------
ABSTAIN_LABEL = "Unknown"  # Unknown 标签统一口径
THRESHOLDS = {"Level2": 0.90, "Level3": 0.90}  # Level2/Level3 的放行阈值
valid_lvl3 = {  # 父子约束（若模型没有聚合类 "UOC"，改为 UOC-H/L/LL）
    "OC": {"EOC-H", "EOC-L", "EOC-LL", "UOC"},
    "CC": {"CM-CO", "CR-clan", "CV"}
}

# -------------------- 小工具函数 --------------------
def apply_threshold(proba: np.ndarray, classes: np.ndarray, thr: float):
    """对分类概率应用阈值：最大概率>=thr 时输出该类别，否则 Unknown。返回(预测, 最大概率)。"""
    max_idx = np.argmax(proba, axis=1)
    max_val = proba[np.arange(proba.shape[0]), max_idx]
    pred = np.where(max_val >= thr, classes[max_idx], ABSTAIN_LABEL)
    return pred, max_val

@st.cache_resource
def load_model_and_metadata():
    """模型与特征列加载：优先 models/ 目录；特征列优先 JSON 退回 model.feature_name_。"""
    def _load(p1, p2): return joblib.load(p1) if os.path.exists(p1) else joblib.load(p2)
    model_lvl1 = _load("models/model_level1.pkl", "model_level1.pkl")
    model_lvl2 = _load("models/model_level2.pkl", "model_level2.pkl")
    model_lvl3 = _load("models/model_level3.pkl", "model_level3.pkl")

    feat_json = "models/feature_columns.json" if os.path.exists("models/feature_columns.json") else "feature_columns.json"
    if os.path.exists(feat_json):
        import json
        with open(feat_json, "r", encoding="utf-8") as f: features = json.load(f)
    else:
        features = getattr(model_lvl1, "feature_name_", None)
        if not features:
            st.error("Feature columns not found (feature_columns.json or model.feature_name_).")
            st.stop()
    return model_lvl1, model_lvl2, model_lvl3, features

@st.cache_resource
def _make_explainer_cached(sig: str, _model):
    """缓存 SHAP explainer；sig 作为缓存键，_model 不参与哈希。"""
    return shap.TreeExplainer(_model)

def _model_signature(model) -> str:
    """构造一个可哈希的模型签名：模型类名 + 排序后的超参 + 类别列表"""
    try:    params_tup = tuple(sorted((k, str(v)) for k, v in model.get_params().items()))
    except Exception: params_tup = ()
    try:    classes = tuple(map(str, getattr(model, "classes_", ())))
    except Exception: classes = ()
    return f"{model.__class__.__name__}|{hash(params_tup)}|{hash(classes)}"

def preprocess_uploaded_data(df):
    """数据预处理：兼容 FeOT 缺失的拆分；派生特征生成（与你原逻辑一致）。"""
    MW = {'TiO2':79.866,'Al2O3':101.961,'Cr2O3':151.99,'FeO':71.844,'MnO':70.937,'MgO':40.304,'ZnO':81.38,'SiO2':60.0843,'V2O3':149.88}
    O_num={'TiO2':2,'Al2O3':3,'Cr2O3':3,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':2,'V2O3':3}
    Cat_num={'TiO2':1,'Al2O3':2,'Cr2O3':2,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':1,'V2O3':2}
    FE2O3_OVER_FEO_FE_EQ = 159.688 / (2 * 71.844)

    for ox in MW:
        if ox not in df.columns: df[ox] = 0.0
    df = df.copy()

    if "FeO" in df.columns and "Fe2O3" in df.columns:
        df = df.rename(columns={"FeO": "FeOre", "Fe2O3": "Fe2O3re"})
        df["FeO_total"] = df["FeOre"] + df["Fe2O3re"] * 0.8998
    else:
        def fe_split_spinel(row, O_basis=32):
            val_feot = 0.0 if pd.isna(row.get('FeOT', np.nan)) else float(row.get('FeOT'))
            moles = {ox: row[ox]/MW[ox] for ox in MW if ox != 'FeO'}
            moles['FeO'] = val_feot / MW['FeO']
            O_total = sum(moles[ox]*O_num[ox] for ox in moles)
            fac = O_basis / O_total if O_total>0 else 0.0
            cations = {ox: moles[ox]*Cat_num[ox]*fac for ox in moles}
            S = sum(cations.values()); T = 24.0
            Fe_total = cations['FeO']
            Fe3 = max(0.0, 2*O_basis*(1 - T/S)) if S>0 else 0.0
            Fe3 = min(Fe3, Fe_total); Fe2 = Fe_total - Fe3
            Fe2_frac = Fe2/Fe_total if Fe_total>0 else 0.0
            Fe3_frac = Fe3/Fe_total if Fe_total>0 else 0.0
            FeO_wt   = Fe2_frac * val_feot
            Fe2O3_wt = Fe3_frac * val_feot * FE2O3_OVER_FEO_FE_EQ
            return pd.Series({'FeOre':FeO_wt,'Fe2O3re':Fe2O3_wt,'Fe2_frac':Fe2_frac,'Fe3_frac':Fe3_frac,'FeO_total':FeO_wt+Fe2O3_wt*0.8998})
        df = df.join(df.apply(fe_split_spinel, axis=1))

    mol_wt = {'Cr2O3':151.99,'Al2O3':101.961,'MgO':40.304,'FeO':71.844,'Fe2O3':159.688}
    Cr_mol = df["Cr2O3"]/mol_wt["Cr2O3"]*2
    Al_mol = df["Al2O3"]/mol_wt["Al2O3"]*2
    Mg_mol = df["MgO"]/mol_wt["MgO"]
    Fe2_mol = df["FeOre"]/mol_wt["FeO"]
    Fe3_mol = df["Fe2O3re"]/mol_wt["Fe2O3"]*2

    df["Cr#"] = Cr_mol / (Cr_mol + Al_mol)
    df["Mg#"] = Mg_mol / (Mg_mol + Fe2_mol)
    df["Fe*"]   = Fe3_mol / (Fe3_mol + Cr_mol + Al_mol)
    df["Fe#"]     = Fe2_mol / (Fe2_mol + Mg_mol)
    return df

def to_numeric_df(df):
    """尽量把所有列转 float，无法转换则置 NaN。"""
    return df.apply(pd.to_numeric, errors="coerce")

# ========= 单层多数票 + 平均概率（含 Unknown & 未路由行）=========
def level_group_stats(labels, classes, prob_by_class, p_max=None, p_unknown=None, fill_unknown_for_empty=True):
    """
    labels: 该层每行的标签（可含空串/Unknown）
    classes: 该层类别数组（顺序与 prob_by_class 列一致）
    prob_by_class: (N, C) 该层“最终用于判定”的每类概率（L3 用父子约束后的 p）
    p_max: 每行该层的最大概率（约束后），用于计算 Unknown 概率
    p_unknown: 若传 None 则用 (1 - p_max)
    fill_unknown_for_empty: True 时，未路由（空串）视为 Unknown，Unknown 概率=1.0，实类概率=0
    返回: (top_label, top_share_str '17/18', top_mean_prob)
    """
    N = len(labels)
    s = pd.Series(labels, dtype="object").fillna("")
    if fill_unknown_for_empty:
        empty_mask = (s == "")
        if empty_mask.any():
            s.loc[empty_mask] = ABSTAIN_LABEL
            if p_unknown is None and p_max is not None:
                p_unknown = np.where(empty_mask, 1.0, (1.0 - (p_max if p_max is not None else 0.0)))
            elif p_unknown is not None:
                p_unknown = np.where(empty_mask, 1.0, p_unknown)
            if prob_by_class is not None and isinstance(prob_by_class, np.ndarray):
                prob_by_class = np.where(
                    np.repeat(empty_mask.values[:, None], prob_by_class.shape[1], axis=1),
                    0.0,
                    np.nan_to_num(prob_by_class, nan=0.0)
                )

    counts = s.value_counts()
    candidates = list(counts.index)

    means = {}
    for lab in candidates:
        if lab == ABSTAIN_LABEL:
            if p_unknown is None and p_max is not None:
                pu = 1.0 - np.array(p_max, dtype=float)
            else:
                pu = np.array(p_unknown, dtype=float) if p_unknown is not None else np.zeros(N)
            means[lab] = float(np.nanmean(pu))
        else:
            if prob_by_class is None:
                means[lab] = 0.0
            else:
                col = np.where(classes == lab)[0]
                if len(col) == 0:
                    means[lab] = 0.0
                else:
                    arr = np.nan_to_num(prob_by_class[:, col[0]], nan=0.0)
                    means[lab] = float(np.mean(arr))

    max_count = counts.max()
    top_cands = [lab for lab, c in counts.items() if c == max_count]
    top_label = max(top_cands, key=lambda lab: means.get(lab, 0.0))
    top_share_str = f"{int(counts[top_label])}/{N}"
    top_mean_prob = means.get(top_label, 0.0)
    return top_label, top_share_str, top_mean_prob

# -------------------- 侧边栏：加载模型 --------------------
with st.sidebar:
    st.subheader("Model Loading")
    try:
        model_lvl1, model_lvl2, model_lvl3, feature_list = load_model_and_metadata()
        st.success("Models and feature list loaded.")
        st.caption(f"Feature dimension: {len(feature_list)}")
    except Exception as e:
        st.error("Failed to load models or feature columns.")
        st.exception(e)

# -------------------- 上传文件并处理 --------------------
uploaded_file = st.file_uploader("Upload an Excel or CSV file (must include all feature columns).", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
        df_uploaded = preprocess_uploaded_data(df_uploaded)

        # 对齐特征列
        df_input = df_uploaded.copy()
        for col in feature_list:
            if col not in df_input.columns: df_input[col] = np.nan
        df_input = to_numeric_df(df_input[feature_list])

        N = len(df_input)

        # ========= Level 1（不启用 Unknown）=========
        prob1 = model_lvl1.predict_proba(df_input)            # (N, C1)
        classes1 = model_lvl1.classes_.astype(str)
        pred1_idx = np.argmax(prob1, axis=1)
        pred1_label = classes1[pred1_idx]
        p1max = prob1[np.arange(N), pred1_idx]                # 仅展示；L1 无 Unknown

        # ========= Level 2（阈值 + Unknown，仅 L1=Extraterrestrial）=========
        _pred1_norm = pd.Series(pred1_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_lvl2 = (_pred1_norm == "extraterrestrial").to_numpy()

        prob2_raw = np.full((N, len(model_lvl2.classes_)), np.nan)
        pred2_label = np.full(N, "", dtype=object)
        p2max = np.full(N, np.nan)
        p2unk = np.full(N, np.nan)

        if mask_lvl2.any():
            pr2 = model_lvl2.predict_proba(df_input[mask_lvl2])
            classes2 = model_lvl2.classes_.astype(str)
            pred2_masked, p2max_masked = apply_threshold(pr2, classes2, THRESHOLDS["Level2"])
            prob2_raw[mask_lvl2] = pr2
            pred2_label[mask_lvl2] = pred2_masked
            p2max[mask_lvl2] = p2max_masked
            p2unk[mask_lvl2] = 1.0 - p2max_masked

        # 对未路由行：视为 Unknown；Unknown 概率=1；实类概率=0
        classes2 = model_lvl2.classes_.astype(str)
        prob2 = np.nan_to_num(prob2_raw, nan=0.0)
        empty2 = (pd.Series(pred2_label, dtype="object") == "")
        if empty2.any():
            pred2_label[empty2.values] = ABSTAIN_LABEL
            p2unk[empty2.values] = 1.0

        # ========= Level 3（父子约束 + 阈值 + Unknown，仅 L2 in {OC,CC}）=========
        _pred2_norm = pd.Series(pred2_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_lvl3 = _pred2_norm.isin(["oc", "cc"]).to_numpy()
        routed_to_L3 = bool(mask_lvl3.any())   # ☆ 关键：整组是否“只到二级”

        C3 = len(model_lvl3.classes_)
        classes3 = model_lvl3.classes_.astype(str)
        prob3_raw = np.full((N, C3), np.nan)
        prob3_post = np.zeros((N, C3))
        pred3_label = np.full(N, "", dtype=object)
        p3max = np.full(N, np.nan)
        p3unk = np.full(N, np.nan)

        if routed_to_L3:
            all_pr3 = model_lvl3.predict_proba(df_input[mask_lvl3])
            idxs = np.where(mask_lvl3)[0]
            prob3_raw[mask_lvl3] = all_pr3
            for row_i, i_global in enumerate(idxs):
                parent = str(pred2_label[i_global])          # "OC" 或 "CC"
                allowed = valid_lvl3.get(parent, set())
                p = all_pr3[row_i].copy()
                if allowed:
                    mask_allowed = np.isin(classes3, list(allowed))
                    p = p * mask_allowed
                    if p.sum() > 0:
                        p = p / p.sum()
                j = int(np.argmax(p)); pmax = float(p[j])
                pred3_label[i_global] = classes3[j] if pmax >= THRESHOLDS["Level3"] else ABSTAIN_LABEL
                p3max[i_global] = pmax
                p3unk[i_global] = 1.0 - pmax
                prob3_post[i_global] = p

            # 未路由到 L3 的行（这时存在）：设为 Unknown，Unknown 概率=1
            empty3 = (pd.Series(pred3_label, dtype="object") == "")
            if empty3.any():
                pred3_label[empty3.values] = ABSTAIN_LABEL
                p3unk[empty3.values] = 1.0

        # -------------------- 构造展示表 --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "Index", df_display.index + 1)
        df_display.insert(1, "Level1_Pred", pred1_label)
        df_display.insert(2, "Level2_Pred", pred2_label)

        # 原始各类概率（便于核查）
        for i, c in enumerate(classes1): df_display[f"P_Level1_{c}"] = prob1[:, i]
        for i, c in enumerate(classes2): df_display[f"P_Level2_{c}"] = prob2[:, i]

        # 只有当整组有样本路由到 L3 时才添加 L3 列
        if routed_to_L3:
            df_display.insert(3, "Level3_Pred", pred3_label)
            for i, c in enumerate(classes3):
                df_display[f"P_Level3_{c}"] = prob3_raw[:, i]  # 原始（未必约束后）

        st.subheader("🧾 Predictions")
        st.dataframe(df_display)

        # -------------------- 组内多数票 + 均值概率（Unknown参与；分母=N） --------------------
        # L1（无 Unknown）
        l1_label, l1_share, l1_mean = level_group_stats(
            labels=pred1_label, classes=classes1, prob_by_class=prob1,
            p_max=p1max, p_unknown=None, fill_unknown_for_empty=False
        )
        # L2（有 Unknown）
        l2_label, l2_share, l2_mean = level_group_stats(
            labels=pred2_label, classes=classes2, prob_by_class=prob2,
            p_max=p2max, p_unknown=p2unk, fill_unknown_for_empty=True
        )
        # L3（只有当 routed_to_L3 为真时才计算）
        if routed_to_L3:
            l3_label, l3_share, l3_mean = level_group_stats(
                labels=pred3_label, classes=classes3, prob_by_class=prob3_post,
                p_max=p3max, p_unknown=p3unk, fill_unknown_for_empty=True
            )

        # 把组级统计写回表（每行相同，便于导出/筛选）
        df_display["L1_TopShare"]    = l1_share
        df_display["L1_TopMeanProb"] = round(l1_mean, 3)
        df_display["L2_TopShare"]    = l2_share
        df_display["L2_TopMeanProb"] = round(l2_mean, 3)
        if routed_to_L3:
            df_display["L3_TopShare"]    = l3_share
            df_display["L3_TopMeanProb"] = round(l3_mean, 3)


       
# -------------------- 📈 SHAP Interpretability --------------------

# -------------------- 📈 SHAP Interpretability --------------------
        # -------------------- 📈 SHAP Interpretability --------------------
        st.subheader("📈 SHAP Interpretability")

        def _safe_class_names(m):   # 取真实类别名并转成字符串
            try:
                return [str(x) for x in list(getattr(m, "classes_", []))]
            except Exception:
                return []

        def _bar_per_class(shap_vals_1class, X, title, top_k=15):
            """
            自己画条形图：按该“类别”的 mean(|SHAP|) 取前 top_k 个特征。
            这样完全避免 shap 内置 legend 里出现 class1/2/3。
            """
            mean_abs = np.mean(np.abs(shap_vals_1class), axis=0)
            order = np.argsort(mean_abs)[-top_k:]                # 取 Top-K
            feats = X.columns.values[order]
            vals  = mean_abs[order]

            fig, ax = plt.subplots(figsize=(6, 3 + 0.2*len(order)))
            ax.barh(range(len(vals)), vals)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(feats)
            ax.set_xlabel("mean |SHAP|")
            ax.set_title(title)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        def _render_shap_for_model(model, level_name, X):
            """
            兼容多分类(list) / 二分类(ndarray)：
            - 多分类：每个“真实类别名”各画一张条形图 + 各画一张蜂群图；
            - 二分类：把 shap ndarray 拆成“负类= -sv、正类= sv”，同样各画一张。
            """
            explainer = _make_explainer_cached(_model_signature(model), _model=model)
            sv = explainer.shap_values(X)
            class_names = _safe_class_names(model)

            # 多分类：sv 是 list[n_classes]
            if isinstance(sv, list):
                # 逐类画：条形图 + 蜂群图（标题写真实类别名）
                for i, cname in enumerate(class_names or [f"class {i}" for i in range(len(sv))]):
                    _bar_per_class(sv[i], X, title=f"{level_name} — class: {cname}")
                    shap.summary_plot(sv[i], X, show=False)
                    plt.title(f"{level_name} — SHAP beeswarm (class: {cname})")
                    st.pyplot(plt.gcf()); plt.close()

            else:
                # 二分类常返回单个 ndarray（正类的 SHAP）
                # 我们手工构造两个类别的视图：负类 = -sv，正类 = sv
                if len(class_names) == 2:
                    sv_list  = [ -sv, sv ]
                    names    = [ class_names[0], class_names[1] ]
                else:
                    # 极少数情况下拿不到 classes_；至少保证正类能画
                    sv_list  = [ sv ]
                    names    = [ class_names[-1] if class_names else "positive" ]

                for arr, cname in zip(sv_list, names):
                    _bar_per_class(arr, X, title=f"{level_name} — class: {cname}")
                    shap.summary_plot(arr, X, show=False)
                    plt.title(f"{level_name} — SHAP beeswarm (class: {cname})")
                    st.pyplot(plt.gcf()); plt.close()

        cols = st.columns(3)
        for col, (mdl, nm) in zip(cols, [(model_lvl1, "Level1"), (model_lvl2, "Level2"), (model_lvl3, "Level3")]):
            with col:
                st.markdown(f"#### 🔍 {nm} Model")
                _render_shap_for_model(mdl, nm, df_input)

        
       

        # -------------------- ✅ 样品一致性 + 组结果（根据是否存在 L3 动态展示） --------------------
        st.subheader("🧪 Specimen Confirmation & Group Result")
        same_specimen = st.checkbox("I confirm all uploaded rows originate from the same physical specimen.")
        if same_specimen:
            depth = {"Level1": 1, "Level2": 2, "Level3": 3}
            cands = [
                ("Level1", {"label": l1_label, "share": int(l1_share.split('/')[0]) / N, "prob": l1_mean,
                            "agree": int(l1_share.split('/')[0]), "total": N}),
                ("Level2", {"label": l2_label, "share": int(l2_share.split('/')[0]) / N, "prob": l2_mean,
                            "agree": int(l2_share.split('/')[0]), "total": N}),
            ]
            if routed_to_L3:
                cands.append(("Level3", {"label": l3_label, "share": int(l3_share.split('/')[0]) / N, "prob": l3_mean,
                                         "agree": int(l3_share.split('/')[0]), "total": N}))
            final_level, final = sorted(cands, key=lambda t: (t[1]["share"], t[1]["prob"], depth[t[0]]), reverse=True)[0]
            st.success(
                f"Final group result → **{final_level}: {final['label']}**  |  "
                f"Probability (mean for this class): **{final['prob']:.3f}**  |  "
                f"Share: **{final['agree']}/{final['total']} ({final['share']:.0%})**"
            )

            # 对比小表
            rows = [
                {"Level": "Level1", "Top class": l1_label, "Share": l1_share, "Mean prob": round(l1_mean, 3)},
                {"Level": "Level2", "Top class": l2_label, "Share": l2_share, "Mean prob": round(l2_mean, 3)},
            ]
            if routed_to_L3:
                rows.append({"Level": "Level3", "Top class": l3_label, "Share": l3_share, "Mean prob": round(l3_mean, 3)})
            st.dataframe(pd.DataFrame(rows))

        # -------------------- 🧩 恢复：训练池 & GitHub 同步 --------------------
        st.subheader("🧩 Add Predictions to Training Pool?")
        if st.checkbox("✅ Confirm to append these samples to the training pool for future retraining"):
            df_save = df_input.copy()
            df_save["Level1"] = pred1_label
            df_save["Level2"] = pred2_label
            if routed_to_L3:
                df_save["Level3"] = pred3_label
            # 也可把组结果写入（如需可解注释）
            # df_save["Group_L1_Top"] = l1_label; df_save["Group_L2_Top"] = l2_label
            # if routed_to_L3: df_save["Group_L3_Top"] = l3_label

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
