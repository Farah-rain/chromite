# streamlit_app.py  —— Chromite Extraterrestrial Origin Classifier (integrated full script)

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os, joblib, requests, base64
from io import BytesIO
from itertools import chain

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="Chromite Extraterrestrial Origin Classifier", layout="wide")
st.title("✨ Chromite Extraterrestrial Origin Classifier")

# -------------------- 常量与映射（与训练一致） --------------------
ABSTAIN_LABEL = "Unclassified"
THRESHOLDS = {"Level2": 0.90, "Level3": 0.90}
MARGINS_LEVEL2 = {"OC": 0.04}

valid_lvl3 = {
    "OC": {"EOC-H", "EOC-L", "EOC-LL", "UOC"},
    "CC": {"CM-CO", "CR-clan", "CV"}
}

# 稳定柔和调色板（用于所有饼图/柱状图）
PALETTE = list(chain(plt.get_cmap("tab20").colors, plt.get_cmap("tab20c").colors))

# -------------------- 概率校准 & 类阈值工具 --------------------
def _load_joblib_pair(primary_path, fallback_path):
    p = primary_path if os.path.exists(primary_path) else fallback_path
    return joblib.load(p) if os.path.exists(p) else None

def load_calibrator_and_threshold(level_name: str):
    calib = _load_joblib_pair(f"models/calib_{level_name}.joblib", f"calib_{level_name}.joblib")
    thr   = _load_joblib_pair(f"models/thr_{level_name}.joblib",   f"thr_{level_name}.joblib")
    return calib, thr

def apply_calibrators(proba: np.ndarray, classes: np.ndarray, calibrators: dict | None):
    if calibrators is None:
        return proba
    P = np.zeros_like(proba, dtype=float)
    for j, cls in enumerate(classes):
        ir = calibrators.get(str(cls)) if isinstance(calibrators, dict) else None
        P[:, j] = (ir.transform(proba[:, j]) if ir is not None else proba[:, j])
    eps = 1e-12
    row_sum = P.sum(axis=1, keepdims=True)
    P = (P + eps) / np.maximum(row_sum + eps * P.shape[1], eps)
    return P

def predict_with_classwise_thresholds(
    proba_cal: np.ndarray,
    classes: np.ndarray,
    thr_dict: dict | None,
    unknown_label: str,
    margins: dict | None = None
):
    C = proba_cal.shape[1]
    thr_dict = thr_dict or {}
    preds, pmax = [], []
    for row in proba_cal:
        cand = [j for j, cls in enumerate(classes) if row[j] >= float(thr_dict.get(str(cls), 0.5))]
        if not cand:
            preds.append(unknown_label); pmax.append(float(np.nanmax(row)))
            continue
        j_best = max(cand, key=lambda k: row[k])
        best_score = row[j_best]
        order = np.argsort(row)[::-1]
        j_second = order[1] if C >= 2 else j_best
        gap = best_score - row[j_second]
        ok_margin = True
        if margins is not None:
            m = float(margins.get(str(classes[j_best]), 0.0))
            ok_margin = (gap >= m)
        if ok_margin:
            preds.append(classes[j_best]); pmax.append(best_score)
        else:
            preds.append(unknown_label); pmax.append(best_score)
    return np.array(preds, dtype=object), np.array(pmax, dtype=float)

# -------------------- 其他工具函数 --------------------
def apply_threshold(proba: np.ndarray, classes: np.ndarray, thr: float):
    max_idx = np.argmax(proba, axis=1)
    max_val = proba[np.arange(proba.shape[0]), max_idx]
    pred = np.where(max_val >= thr, classes[max_idx], ABSTAIN_LABEL)
    return pred, max_val

@st.cache_resource
def load_model_and_metadata():
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
    return shap.TreeExplainer(_model)

def _model_signature(model) -> str:
    try:    params_tup = tuple(sorted((k, str(v)) for k, v in model.get_params().items()))
    except Exception: params_tup = ()
    try:    classes = tuple(map(str, getattr(model, "classes_", ())))
    except Exception: classes = ()
    return f"{model.__class__.__name__}|{hash(params_tup)}|{hash(classes)}"

def preprocess_uploaded_data(df):
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
    df["Fe*"] = Fe3_mol / (Fe3_mol + Cr_mol + Al_mol)
    df["Fe#"] = Fe2_mol / (Fe2_mol + Mg_mol)
    return df

def to_numeric_df(df):
    return df.apply(pd.to_numeric, errors="coerce")

# ========= 单层多数票 + 平均概率（含 Unclassified & 未路由行）=========
def level_group_stats(labels, classes, prob_by_class, p_max=None, p_unknown=None, fill_unknown_for_empty=True):
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

# 载入校准器 & 类阈值（若存在）
calib_L2, thr_L2 = load_calibrator_and_threshold("Level2")
calib_L3, thr_L3 = load_calibrator_and_threshold("Level3")

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

        # ========= Level 1 =========
        prob1 = model_lvl1.predict_proba(df_input)
        classes1 = model_lvl1.classes_.astype(str)
        pred1_idx = np.argmax(prob1, axis=1)
        pred1_label = classes1[pred1_idx]
        p1max = prob1[np.arange(N), pred1_idx]

        # ========= Level 2（只对 L1=Extraterrestrial）=========
        _pred1_norm = pd.Series(pred1_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_lvl2 = (_pred1_norm == "extraterrestrial").to_numpy()

        prob2_raw = np.full((N, len(model_lvl2.classes_)), np.nan)
        pred2_label = np.full(N, "", dtype=object)
        p2max = np.full(N, np.nan)
        p2unk = np.full(N, np.nan)
        classes2 = model_lvl2.classes_.astype(str)

        if mask_lvl2.any():
            pr2 = model_lvl2.predict_proba(df_input[mask_lvl2])
            pr2_cal = apply_calibrators(pr2, classes2, calib_L2)
            if thr_L2 is not None:
                pred2_masked, p2max_masked = predict_with_classwise_thresholds(
                    proba_cal=pr2_cal, classes=classes2, thr_dict=thr_L2,
                    unknown_label=ABSTAIN_LABEL, margins=MARGINS_LEVEL2
                )
            else:
                pred2_masked, p2max_masked = apply_threshold(pr2_cal, classes2, THRESHOLDS["Level2"])

            prob2_raw[mask_lvl2] = pr2
            pred2_label[mask_lvl2] = pred2_masked
            p2max[mask_lvl2] = p2max_masked
            p2unk[mask_lvl2] = 1.0 - p2max_masked

        prob2 = np.nan_to_num(prob2_raw, nan=0.0)
        empty2 = (pd.Series(pred2_label, dtype="object") == "")
        if empty2.any():
            pred2_label[empty2.values] = ABSTAIN_LABEL
            p2unk[empty2.values] = 1.0

        # ========= Level 3（父子约束）=========
        _pred2_norm = pd.Series(pred2_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_lvl3 = _pred2_norm.isin(["oc", "cc"]).to_numpy()
        routed_to_L3 = bool(mask_lvl3.any())

        C3 = len(model_lvl3.classes_)
        classes3 = model_lvl3.classes_.astype(str)
        prob3_raw = np.full((N, C3), np.nan)
        prob3_post = np.zeros((N, C3))
        pred3_label = np.full(N, "", dtype=object)
        p3max = np.full(N, np.nan)
        p3unk = np.full(N, np.nan)

        if routed_to_L3:
            all_pr3 = model_lvl3.predict_proba(df_input[mask_lvl3])
            all_pr3_cal = apply_calibrators(all_pr3, classes3, calib_L3)

            idxs = np.where(mask_lvl3)[0]
            prob3_raw[mask_lvl3] = all_pr3
            for row_i, i_global in enumerate(idxs):
                parent = str(pred2_label[i_global])
                allowed = valid_lvl3.get(parent, set())
                p = all_pr3_cal[row_i].copy()
                if allowed:
                    mask_allowed = np.isin(classes3, list(allowed))
                    p = p * mask_allowed
                    s = p.sum()
                    if s > 0: p = p / s
                if thr_L3 is not None:
                    pred_tmp, pmax_tmp = predict_with_classwise_thresholds(
                        proba_cal=p.reshape(1, -1),
                        classes=classes3,
                        thr_dict=thr_L3,
                        unknown_label=ABSTAIN_LABEL,
                        margins=None
                    )
                    pred3_label[i_global] = pred_tmp[0]
                    p3max[i_global] = pmax_tmp[0]
                else:
                    j = int(np.argmax(p)); pmax = float(p[j])
                    pred3_label[i_global] = classes3[j] if pmax >= THRESHOLDS["Level3"] else ABSTAIN_LABEL
                    p3max[i_global] = pmax
                p3unk[i_global] = 1.0 - p3max[i_global]
                prob3_post[i_global] = p

            empty3 = (pd.Series(pred3_label, dtype="object") == "")
            if empty3.any():
                pred3_label[empty3.values] = ABSTAIN_LABEL
                p3unk[empty3.values] = 1.0

        # -------------------- 结果表 --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "Index", df_display.index + 1)
        df_display.insert(1, "Level1_Pred", pred1_label)
        df_display.insert(2, "Level2_Pred", pred2_label)
        for i, c in enumerate(classes1): df_display[f"P_Level1_{c}"] = prob1[:, i]
        for i, c in enumerate(classes2): df_display[f"P_Level2_{c}"] = prob2[:, i]
        if routed_to_L3:
            df_display.insert(3, "Level3_Pred", pred3_label)
            for i, c in enumerate(classes3):
                df_display[f"P_Level3_{c}"] = prob3_raw[:, i]

        st.subheader("🧾 Predictions")
        st.dataframe(df_display, use_container_width=True)

        # -------------------- 组内多数票 + 均值概率 --------------------
        l1_label, l1_share, l1_mean = level_group_stats(
            labels=pred1_label, classes=classes1, prob_by_class=prob1,
            p_max=p1max, p_unknown=None, fill_unknown_for_empty=False
        )
        l2_label, l2_share, l2_mean = level_group_stats(
            labels=pred2_label, classes=classes2, prob_by_class=prob2,
            p_max=p2max, p_unknown=p2unk, fill_unknown_for_empty=True
        )
        if routed_to_L3:
            l3_label, l3_share, l3_mean = level_group_stats(
                labels=pred3_label, classes=classes3, prob_by_class=prob3_post,
                p_max=p3max, p_unknown=p3unk, fill_unknown_for_empty=True
            )

        df_display["L1_TopShare"]    = l1_share
        df_display["L1_TopMeanProb"] = round(l1_mean, 3)
        df_display["L2_TopShare"]    = l2_share
        df_display["L2_TopMeanProb"] = round(l2_mean, 3)
        if routed_to_L3:
            df_display["L3_TopShare"]    = l3_share
            df_display["L3_TopMeanProb"] = round(l3_mean, 3)

        # -------------------- SHAP：tabs 可横向滚动 + 三列并排 --------------------
        st.subheader("📈 SHAP Interpretability")
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"]{
            overflow-x:auto!important;overflow-y:hidden;white-space:nowrap;
            scrollbar-width:thin;-ms-overflow-style:auto;
        }
        .stTabs [data-baseweb="tab"]{white-space:nowrap;padding:6px 10px;margin:0 2px;}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{ height:8px; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb{ background:rgba(0,0,0,.25); border-radius:8px; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track{ background:rgba(0,0,0,.06); border-radius:8px; }
        </style>
        """, unsafe_allow_html=True)

        TOP_K = 13
        chart_kind = st.radio("Per-class SHAP view", ["Bar (mean |SHAP|)", "Beeswarm"], horizontal=True, index=0)

        def _safe_class_names(m):
            try:
                return [str(x) for x in list(getattr(m, "classes_", []))]
            except Exception:
                return []

        def _bar_per_class(shap_vals_1class, X, title, top_k=TOP_K):
            mean_abs = np.mean(np.abs(shap_vals_1class), axis=0).reshape(-1)
            order = np.argsort(mean_abs)
            k = min(top_k, len(order))
            sel = order[-k:]
            feats = np.array(X.columns)[sel]
            vals  = mean_abs[sel]
            fig, ax = plt.subplots(figsize=(7, max(3, 0.28*len(sel)+2)))
            ax.barh(np.arange(len(vals)), vals)
            ax.set_yticks(np.arange(len(vals)))
            ax.set_yticklabels(feats)
            ax.set_xlabel("mean |SHAP|")
            ax.set_title(title)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        def _sv_to_list_per_class(sv, X, class_names):
            N, F = X.shape
            if isinstance(sv, list):
                return [np.asarray(a).reshape(N, F) for a in sv]
            arr = np.asarray(sv)
            if arr.ndim == 2:
                r, c = arr.shape
                if r == N and c == F:
                    if class_names and len(class_names) == 2: return [-arr, arr]
                    return [arr]
                if r == N and c % F == 0:
                    C = c // F; return [arr[:, i*F:(i+1)*F].reshape(N, F) for i in range(C)]
                if c == F and r % N == 0:
                    C = r // N; return [arr[i*N:(i+1)*N, :].reshape(N, F) for i in range(C)]
                if class_names and arr.size == N*F*len(class_names):
                    C = len(class_names)
                    try:    tmp = arr.reshape(N, F, C); return [tmp[:, :, i] for i in range(C)]
                    except: 
                        try: tmp = arr.reshape(C, N, F); return [tmp[i, :, :] for i in range(C)]
                        except: pass
                return [arr.reshape(N, F)]
            if arr.ndim == 3:
                if arr.shape[0] == N and arr.shape[1] == F:
                    C = arr.shape[2]; return [arr[:, :, i].reshape(N, F) for i in range(C)]
                if arr.shape[1] == N and arr.shape[2] == F:
                    C = arr.shape[0]; return [arr[i, :, :].reshape(N, F) for i in range(C)]
                if arr.shape[0] == N and arr.shape[2] == F:
                    C = arr.shape[1]; return [arr[:, i, :].reshape(N, F) for i in range(C)]
            return [arr.reshape(N, F)]

        def _render_per_class(model, level_name, X):
            explainer = _make_explainer_cached(_model_signature(model), _model=model)
            raw_sv = explainer.shap_values(X)
            class_names = _safe_class_names(model)
            sv_list = _sv_to_list_per_class(raw_sv, X, class_names)
            if not class_names or len(class_names) != len(sv_list):
                class_names = [f"class {i}" for i in range(len(sv_list))]
                if len(sv_list) == 2: class_names = ["negative", "positive"]
            tabs = st.tabs(class_names)
            for tab, cname, arr in zip(tabs, class_names, sv_list):
                with tab:
                    if chart_kind.startswith("Bar"):
                        _bar_per_class(arr, X, title=f"{level_name} · {cname}", top_k=TOP_K)
                    else:
                        shap.summary_plot(arr, X, max_display=TOP_K, show=False)
                        plt.title(f"{level_name} · {cname}")
                        st.pyplot(plt.gcf()); plt.close()

        cols = st.columns(3)
        for col, (mdl, nm) in zip(cols, [(model_lvl1, "Level1"), (model_lvl2, "Level2"), (model_lvl3, "Level3")]):
            with col:
                st.markdown(f"#### 🔍 {nm} (per class)")
                _render_per_class(mdl, nm, df_input)

        # -------------------- Summary + 饼图 + 柱状图 + 训练池 + 下载（整合块） --------------------

        # —— 每层计数（按真实类别名）——
        def _vc_df_from_labels(labels: np.ndarray) -> pd.DataFrame:
            s = pd.Series(labels, dtype="object").fillna(ABSTAIN_LABEL).replace("", ABSTAIN_LABEL)
            vc = s.value_counts(dropna=False)
            df = vc.rename_axis("Class").reset_index(name="count")
            df["share"] = (df["count"] / float(len(s) if len(s) else 1)).round(3)
            return df[["Class", "count", "share"]]

        df_l1 = _vc_df_from_labels(pred1_label).sort_values(["count","Class"], ascending=[False,True], ignore_index=True)
        df_l2 = _vc_df_from_labels(pred2_label).sort_values(["count","Class"], ascending=[False,True], ignore_index=True)
        df_l3 = (_vc_df_from_labels(pred3_label).sort_values(["count","Class"], ascending=[False,True], ignore_index=True)
                 if routed_to_L3 else pd.DataFrame(columns=["Class","count","share"]))

        # —— 合并 Others（小类折叠）——
        def _collapse_others(df: pd.DataFrame, total_n: int, keep_top: int = 7, tiny_cut: float = 0.02) -> pd.DataFrame:
            if df.empty: return df
            df = df.sort_values(["count","Class"], ascending=[False,True]).reset_index(drop=True)
            if total_n <= 0 or len(df) <= keep_top:
                out = df.copy()
            else:
                frac = df["count"] / float(total_n)
                mask_tiny = (frac < tiny_cut)
                head = df.loc[~mask_tiny].head(max(keep_top-1, 0))
                tail = pd.concat([df.loc[mask_tiny], df.loc[~mask_tiny].iloc[max(keep_top-1, 0):]], ignore_index=True)
                if len(tail) > 0:
                    others = pd.DataFrame([{
                        "Class": "Others",
                        "count": int(tail["count"].sum()),
                        "share": round(float(tail["count"].sum()) / float(total_n) if total_n>0 else 0.0, 3)
                    }])
                    out = pd.concat([head, others], ignore_index=True)
                else:
                    out = head.copy()
            s = float(out["count"].sum()) or 1.0
            out["share"] = (out["count"] / s).round(3)
            return out

        def _pie_full(col, df: pd.DataFrame, title: str, total_n: int,
                    small_cut: float = 0.06, tiny_cut: float = 0.02):
            """完整圆饼（非环形、不卡位、不引线、不错开），小扇区自动不显示百分比，右侧图例。"""
            with col:
                if df.empty or int(df["count"].sum()) == 0 or total_n == 0:
                    fig, ax = plt.subplots(figsize=(4.8,4.2))
                    ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.axis("off")
                    st.pyplot(fig); plt.close(fig); return

                # 折叠“Others”（把极小类合并，减少过多扇区；阈值可调）
                def _collapse_others(df_in: pd.DataFrame, keep_top=8, tiny=0.02):
                    df_in = df_in.sort_values(["count","Class"], ascending=[False,True]).reset_index(drop=True)
                    if len(df_in) <= keep_top: 
                        return df_in
                    frac = df_in["count"] / float(total_n)
                    head = df_in.loc[frac >= tiny].head(keep_top-1)
                    tail = pd.concat([df_in.loc[frac < tiny], df_in.loc[frac >= tiny].iloc[max(keep_top-1,0):]])
                    if len(tail) > 0:
                        others = pd.DataFrame([{
                            "Class": "Others",
                            "count": int(tail["count"].sum()),
                            "share": round(float(tail["count"].sum())/float(total_n), 3)
                        }])
                        out = pd.concat([head, others], ignore_index=True)
                    else:
                        out = head
                    s = float(out["count"].sum()) or 1.0
                    out["share"] = (out["count"]/s).round(3)
                    return out

                df_plot = _collapse_others(df, keep_top=8, tiny=tiny_cut)

                labels = df_plot["Class"].astype(str).tolist()
                sizes  = df_plot["count"].astype(int).to_numpy()
                fracs  = sizes / sizes.sum()

                colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

                # 小于 small_cut 的扇区不显示百分比，避免挤作一团
                def _autopct(p):
                    return f"{p:.0f}%" if (p/100.0) >= small_cut else ""

                fig, ax = plt.subplots(figsize=(6.2, 4.8))
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    startangle=110, counterclock=False,
                    colors=colors,
                    labels=None,                        # 标签统一放图例
                    autopct=_autopct,                   # 百分比只给“较大”扇区
                    pctdistance=0.72,                   # 百分比文本靠内一点，减少重叠
                    labeldistance=1.10,                 #（虽未显示labels，留默认布局）
                    wedgeprops=dict(linewidth=0.9, edgecolor="white")  # 更清爽的边线
                )

                # 右侧图例（不再画引线，避免交叉）
                ax.legend(wedges, labels, title="Class", loc="center left",
                        bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

                ax.axis("equal")
                ax.set_title(title, pad=10)
                st.pyplot(fig); plt.close(fig)

        

        # —— 类别频率柱状图 —— 
        def _bar_from_df(col, df: pd.DataFrame, title: str):
            with col:
                if df.empty or int(df["count"].sum()) == 0:
                    st.info("No data"); return
                fig, ax = plt.subplots(figsize=(6.0,3.8))
                x = df["Class"].astype(str).tolist()
                y = df["count"].astype(int).tolist()
                ax.bar(range(len(x)), y, edgecolor="black", color=[PALETTE[i % len(PALETTE)] for i in range(len(x))])
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels(x, rotation=35, ha="right")
                ax.set_ylabel("Count"); ax.set_title(title)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # —— 三列环形饼图 —— 
       
        _pie_full(cols_pie[0], df_l1, "Level1 · class share", total_n=N)
        _pie_full(cols_pie[1], df_l2, "Level2 · class share", total_n=N)
        _pie_full(cols_pie[2], df_l3, "Level3 · class share", total_n=N)


        # —— 三列频率柱状图 —— 
        st.subheader("Class frequency (bars)")
        cols_bar = st.columns(3, gap="large")
        _bar_from_df(cols_bar[0], df_l1, "Level1 · frequency")
        _bar_from_df(cols_bar[1], df_l2, "Level2 · frequency")
        _bar_from_df(cols_bar[2], df_l3, "Level3 · frequency")

        # —— 训练池（放在柱状图之后，下载之前） —— 
        st.subheader("🧩 Add Predictions to Training Pool?")
        if st.checkbox("✅ Confirm to append these samples to the training pool for future retraining"):
            df_save = df_input.copy()
            df_save["Level1"] = pred1_label
            df_save["Level2"] = pred2_label
            if routed_to_L3:
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

        # —— 下载（Prediction + Summary） —— 
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Prediction')
            df_l1_export = df_l1.copy(); df_l1_export.insert(0, "Level", "Level1")
            df_l2_export = df_l2.copy(); df_l2_export.insert(0, "Level", "Level2")
            df_l1_export.to_excel(writer, index=False, sheet_name='Summary_L1')
            df_l2_export.to_excel(writer, index=False, sheet_name='Summary_L2')
            if not df_l3.empty:
                df_l3_export = df_l3.copy(); df_l3_export.insert(0, "Level", "Level3")
                df_l3_export.to_excel(writer, index=False, sheet_name='Summary_L3')

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
