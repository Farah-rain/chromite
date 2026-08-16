import streamlit as st
import streamlit.components.v1 as components
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

# -------------------- 网页字体 --------------------
st.markdown("""
<style>

/* 普通网页控件：明显大一点 */
.stMarkdown p,
[data-testid="stCaptionContainer"],
[data-testid="stCheckbox"] label,
.stButton button,
.stDownloadButton button,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
.stTabs [data-baseweb="tab"],
.stRadio label,
.stRadio [role="radiogroup"] label p {
    font-size: 21px !important;
}

/* Level1 / Level2 (per class) */
div[data-testid="stMarkdownContainer"] h4 {
    font-size: 26px !important;
}

/* 上传框内部稍大；上传框上方说明保持较小 */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    font-size: 20px !important;
}
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] > label p {
    font-size: 18px !important;
}


/* larger checkbox labels */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] span {
    font-size: 21px !important;
}

/* larger download / normal buttons */
.stDownloadButton button,
.stDownloadButton button p,
.stButton button,
.stButton button p {
    font-size: 20px !important;
}


[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] button p {
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- 常量与映射（与训练一致） --------------------
ABSTAIN_LABEL = "Unclassified"

# -------------------- 网站显示标签（只改变显示，不改变旧模型内部标签） --------------------
LEVEL1_DISPLAY_MAP = {
    "terrestrial": "terrestrial",
    "extraterrestrial": "extraterrestrial",
}

LEVEL2_DISPLAY_MAP = {
    "oc": "equilibrated ordinary chondrites (EOC)",
    "eoc": "equilibrated ordinary chondrites (EOC)",
    "cc": "carbonaceous chondrites (CC)",
    "a-l": "acapulcoites-lodranites (A-L)",
    "mars": "martian meteorites",
    "martian": "martian meteorites",
    "martian meteorites": "martian meteorites",
    "brachinite": "brachinites",
    "brachinites": "brachinites",
    "low-ti lunar": "low-Ti lunar samples",
    "low-ti lunar sample": "low-Ti lunar samples",
    "low-ti lunar samples": "low-Ti lunar samples",
    "high-ti lunar": "high-Ti lunar samples",
    "high-ti lunar sample": "high-Ti lunar samples",
    "high-ti lunar samples": "high-Ti lunar samples",
    "win-iab": "winonaite-IAB irons (Win-IAB)",
    "win–iab": "winonaite-IAB irons (Win-IAB)",
    "win-iab iron": "winonaite-IAB irons (Win-IAB)",
    "win-iab irons": "winonaite-IAB irons (Win-IAB)",
    "hed-mes": "HED-mesosiderites (HED-Mes)",
    "hed-mesosiderite": "HED-mesosiderites (HED-Mes)",
    "hed-mesosiderites": "HED-mesosiderites (HED-Mes)",
    "merge": "HED-mesosiderites (HED-Mes)",
    "pallasite": "pallasites",
    "pallasites": "pallasites",
    "unclassified": "Unclassified",
}

def display_level1_label(label):
    if pd.isna(label):
        return label
    raw = str(label).strip()
    return LEVEL1_DISPLAY_MAP.get(raw.casefold(), raw)

def display_level2_label(label):
    if pd.isna(label):
        return label
    raw = str(label).strip()
    return LEVEL2_DISPLAY_MAP.get(raw.casefold(), raw)

def display_level1_array(labels):
    return np.array([display_level1_label(x) for x in labels], dtype=object)

def display_level2_array(labels):
    return np.array([display_level2_label(x) for x in labels], dtype=object)



# 柔和调色板（饼图/柱状图通用）
PALETTE = list(chain(plt.get_cmap("tab20").colors, plt.get_cmap("tab20c").colors))

# -------------------- 右侧控制（字体/尺寸缩放 A±） --------------------
with st.sidebar:
    st.subheader("Display / Models")
    chart_scale = st.slider("Chart scale (A±)", 0.65, 1.20, 0.80, 0.05)
    st.caption("Build: chart-shortlabels-table3dp-v21")

    
    def load_model_and_metadata():
        def _load(p1, p2): return joblib.load(p1) if os.path.exists(p1) else joblib.load(p2)
        model_lvl1 = _load("models/model_level1.pkl", "model_level1.pkl")
        model_lvl2 = _load("models/model_level2.pkl", "model_level2.pkl")

        feat_json = "models/feature_columns.json" if os.path.exists("models/feature_columns.json") else "feature_columns.json"
        if os.path.exists(feat_json):
            import json
            with open(feat_json, "r", encoding="utf-8") as f: features = json.load(f)
        else:
            features = getattr(model_lvl1, "feature_name_", None)
            if not features:
                st.error("Feature columns not found (feature_columns.json or model.feature_name_).")
                st.stop()
        return model_lvl1, model_lvl2, features

    try:
        model_lvl1, model_lvl2, feature_list = load_model_and_metadata()
        st.success("Models and feature list loaded.")
        st.caption(f"Feature dimension: {len(feature_list)}")
    except Exception as e:
        st.error("Failed to load models or feature columns.")
        st.exception(e)

# 载入校准器 & 类阈值（若存在）
def _load_joblib_pair(primary_path, fallback_path):
    p = primary_path if os.path.exists(primary_path) else fallback_path
    return joblib.load(p) if os.path.exists(p) else None

def load_calibrator_and_threshold(level_name: str):
    calib = _load_joblib_pair(f"models/calib_{level_name}.joblib", f"calib_{level_name}.joblib")
    thr   = _load_joblib_pair(f"models/thr_{level_name}.joblib",   f"thr_{level_name}.joblib")
    return calib, thr

calib_L1, thr_L1 = load_calibrator_and_threshold("Level1")

calib_L2, thr_L2 = load_calibrator_and_threshold("Level2")
# ✅ NEW: 载入 KNNImputer（训练阶段保存的）
imp_L1 = _load_joblib_pair("models/knn_imputer_Level1.joblib", "knn_imputer_Level1.joblib")
imp_L2 = _load_joblib_pair("models/knn_imputer_Level2.joblib", "knn_imputer_Level2.joblib")
def _load_imputer_cols(level_name: str):
    p1 = f"models/knn_imputer_{level_name}_columns.json"
    p2 = f"knn_imputer_{level_name}_columns.json"
    p = p1 if os.path.exists(p1) else p2
    if os.path.exists(p):
        import json
        with open(p, "r", encoding="utf-8") as f:
            cols = json.load(f)
        return [str(c) for c in cols]
    return None

def _expected_n_features(imp):
    if imp is None:
        return None
    if hasattr(imp, "n_features_in_"):
        try:
            return int(getattr(imp, "n_features_in_"))
        except Exception:
            pass
    if hasattr(imp, "statistics_"):
        try:
            return int(len(getattr(imp, "statistics_")))
        except Exception:
            pass
    return None

def _apply_imputer_strict(dfX: pd.DataFrame, imp, level_name: str, fallback_cols: list[str]):
    """
    1) 优先用训练时保存的 columns.json 来固定列顺序
    2) 若没有 columns.json，则用 feature_list（fallback_cols）
    3) 若列数与 imputer 期望不一致 -> 直接 stop（避免错位）
    4) 若 imp=None -> 用上传数据临时 fit 一个 KNNImputer 兜底（会提示）
    """
    # 目标列顺序：优先 columns.json，其次 feature_list
    cols = _load_imputer_cols(level_name) or list(fallback_cols)

    # 对齐列：reindex 会自动补缺失列为 NaN、丢弃多余列，并按 cols 排序
    dfX_aligned = dfX.reindex(columns=cols)
    dfX_aligned = dfX_aligned.apply(pd.to_numeric, errors="coerce")

    # 没有 imputer -> 临时兜底（不崩，但会提示）
    if imp is None:
        st.warning(f"⚠️ {level_name}: saved KNNImputer not found. Using a temporary KNNImputer fitted on uploaded data (results may be less stable).")
        from sklearn.impute import KNNImputer
        imp_tmp = KNNImputer(n_neighbors=5, weights="distance")
        arr = imp_tmp.fit_transform(dfX_aligned)
        return pd.DataFrame(arr, columns=cols, index=dfX.index)

    # 有 imputer -> 检查列数一致性（不一致就停）
    exp_n = _expected_n_features(imp)
    if exp_n is not None and exp_n != len(cols):
        st.error(
            f"❌ {level_name}: imputer expects {exp_n} features, but current aligned columns = {len(cols)}.\n"
            f"    This usually means your website feature_list / columns.json does NOT match training.\n"
            f"    Please regenerate and upload matching knn_imputer_{level_name}_columns.json (recommended) "
            f"or ensure feature_list is identical to training."
        )
        st.stop()

    # 正常 transform
    arr = imp.transform(dfX_aligned)
    return pd.DataFrame(arr, columns=cols, index=dfX.index)
# 载入 Tukey 区间（若存在）
#q_low_L2  = _load_joblib_pair("models/q_low_Level2.joblib",  "q_low_Level2.joblib")
#q_high_L2 = _load_joblib_pair("models/q_high_Level2.joblib", "q_high_Level2.joblib")


# -------------------- 概率校准 & 类阈值工具 --------------------
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
            preds.append(unknown_label); pmax.append(float(np.nanmax(row))); continue
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


def _make_explainer_cached(sig: str, _model):
    return shap.TreeExplainer(_model)

def _model_signature(model) -> str:
    try:    params_tup = tuple(sorted((k, str(v)) for k, v in model.get_params().items()))
    except Exception: params_tup = ()
    try:    classes = tuple(map(str, getattr(model, "classes_", ())))
    except Exception: classes = ()
    return f"{model.__class__.__name__}|{hash(params_tup)}|{hash(classes)}"

# 结果下载通用
def _save_fig_as_png_bytes(fig, dpi=220):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _short_chart_label(label):
    """Use the abbreviation inside the last parentheses when available."""
    raw = str(label).strip()
    if raw == "Others":
        return raw
    if "(" in raw and raw.endswith(")"):
        i = raw.rfind("(")
        j = raw.rfind(")")
        if i != -1 and j > i + 1:
            return raw[i + 1:j].strip()
    return raw


def _round_float_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round floating-point columns while preserving integer/text columns."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _format_table_for_html(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Format float columns to fixed decimals for website display."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.{decimals}f}"
            )
    return out


def render_big_scroll_table(df: pd.DataFrame, height: int = 430, font_px: int = 21):
    """Render a real scrollable HTML table with controllable font size."""
    if df is None or df.empty:
        st.info("No data")
        return

    df_html = _format_table_for_html(df, decimals=3)

    html_table = df_html.to_html(
        index=False,
        escape=True,
        border=0,
        classes="big-scroll-table"
    )

    html_doc = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: white;
            font-family: Arial, sans-serif;
        }}

        .big-table-wrap {{
            width: 100%;
            height: {height}px;
            overflow: auto;
            border: 1px solid #e5e7eb;
            border-radius: 7px;
            box-sizing: border-box;
            background: white;
        }}

        table.big-scroll-table {{
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            font-size: {font_px}px;
            line-height: 1.35;
        }}

        table.big-scroll-table th {{
            position: sticky;
            top: 0;
            z-index: 3;
            background: #f6f7f9;
            font-size: {font_px}px;
            font-weight: 600;
            color: #30343b;
            white-space: nowrap;
            padding: 9px 12px;
            border-bottom: 1px solid #d9dde3;
            border-right: 1px solid #eceff3;
            text-align: left;
        }}

        table.big-scroll-table td {{
            font-size: {font_px}px;
            color: #30343b;
            white-space: nowrap;
            padding: 9px 12px;
            border-bottom: 1px solid #eceff3;
            border-right: 1px solid #f1f3f5;
            text-align: left;
        }}

        table.big-scroll-table tr:nth-child(even) td {{
            background: #fbfbfc;
        }}
    </style>
    </head>
    <body>
        <div class="big-table-wrap">
            {html_table}
        </div>
    </body>
    </html>
    """

    # Use a real HTML component; st.markdown can expose/flatten large table HTML.
    components.html(
        html_doc,
        height=height + 8,
        scrolling=False
    )



# -------------------- 数据预处理 --------------------

def preprocess_uploaded_data(df):
    df = df.copy()

    MW = {'TiO2':79.866,'Al2O3':101.961,'Cr2O3':151.99,'FeO':71.844,
          'MnO':70.937,'MgO':40.304,'ZnO':81.38,'SiO2':60.0843,'V2O3':149.88}
    O_num  = {'TiO2':2,'Al2O3':3,'Cr2O3':3,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':2,'V2O3':3}
    Cat_num= {'TiO2':1,'Al2O3':2,'Cr2O3':2,'FeO':1,'MnO':1,'MgO':1,'ZnO':1,'SiO2':1,'V2O3':2}
    FE2O3_OVER_FEO_FE_EQ = 159.688 / (2 * 71.844)

    # 记住原始列名
    orig_cols = set(df.columns)

    # ===== 情况 1：用户已经分开给了 FeO 和 Fe2O3 =====
    if ("FeO" in orig_cols) and ("Fe2O3" in orig_cols):
        # 确保是数值
        df["FeO"]   = pd.to_numeric(df["FeO"],   errors="coerce")
        df["Fe2O3"] = pd.to_numeric(df["Fe2O3"], errors="coerce")

        # 为后续 Fe* / Fe# 计算建一份带 “re” 后缀的复制列
        df["FeOre"]   = df["FeO"]
        df["Fe2O3re"] = df["Fe2O3"]

        df["FeO_total"] = df["FeOre"] + df["Fe2O3re"] * 0.8998

    # ===== 情况 2：只有 FeOT，需要从 FeOT 拆成 FeO + Fe2O3 =====
    else:
        def fe_split_spinel(row, O_basis=32):
            # 把各种 None/"None"/""/NaN 统一转成 0.0（仅用于 FeOT 拆分这一步）
            def _num0(v):
                v = pd.to_numeric(v, errors="coerce")
                return 0.0 if pd.isna(v) else float(v)

            feot = pd.to_numeric(row.get("FeOT", np.nan), errors="coerce")
            if pd.isna(feot):
                # 没 FeOT 就没法拆
                return pd.Series({
                    "FeOre": np.nan, "Fe2O3re": np.nan,
                    "Fe2_frac": np.nan, "Fe3_frac": np.nan,
                    "FeO_total": np.nan
                })

            # 其它氧化物：缺失按 0 参与（否则 O_total 会 NaN 直接全崩）
            wt = {ox: _num0(row.get(ox, 0.0)) for ox in MW if ox != "FeO"}

            moles = {ox: wt[ox] / MW[ox] for ox in wt}
            moles["FeO"] = float(feot) / MW["FeO"]

            O_total = sum(moles[ox] * O_num[ox] for ox in moles)
            if (not np.isfinite(O_total)) or (O_total <= 0):
                return pd.Series({
                    "FeOre": np.nan, "Fe2O3re": np.nan,
                    "Fe2_frac": np.nan, "Fe3_frac": np.nan,
                    "FeO_total": np.nan
                })

            fac = O_basis / O_total
            cations = {ox: moles[ox] * Cat_num[ox] * fac for ox in moles}
            S = sum(cations.values())
            T = 24.0

            if (not np.isfinite(S)) or (S <= 0) or (not np.isfinite(cations["FeO"])) or (cations["FeO"] <= 0):
                return pd.Series({
                    "FeOre": np.nan, "Fe2O3re": np.nan,
                    "Fe2_frac": np.nan, "Fe3_frac": np.nan,
                    "FeO_total": np.nan
                })

            Fe_total = cations["FeO"]
            Fe3 = max(0.0, 2 * O_basis * (1 - T / S))
            Fe3 = min(Fe3, Fe_total)
            Fe2 = Fe_total - Fe3

            Fe2_frac = Fe2 / Fe_total if Fe_total > 0 else np.nan
            Fe3_frac = Fe3 / Fe_total if Fe_total > 0 else np.nan

            FeO_wt = Fe2_frac * float(feot)
            Fe2O3_wt = Fe3_frac * float(feot) * FE2O3_OVER_FEO_FE_EQ

            return pd.Series({
                "FeOre": FeO_wt,
                "Fe2O3re": Fe2O3_wt,
                "Fe2_frac": Fe2_frac,
                "Fe3_frac": Fe3_frac,
                "FeO_total": FeO_wt + Fe2O3_wt * 0.8998
            })

        extra = df.apply(fe_split_spinel, axis=1)
        df = df.join(extra)

        
        df["FeO"]   = df["FeOre"]
        df["Fe2O3"] = df["Fe2O3re"]
        
    den = (df["TiO2"] + df["Al2O3"] + df["Cr2O3"])
    df["TAC"] = np.where(den > 0, df["TiO2"] / den, np.nan)
  
    for ox in MW:
        if ox not in df.columns:
            df[ox] = np.nan

   
    mol_wt = {'Cr2O3':151.99,'Al2O3':101.961,'MgO':40.304,'FeO':71.844,'Fe2O3':159.688}
    Cr_mol = df["Cr2O3"]/mol_wt["Cr2O3"]*2
    Al_mol = df["Al2O3"]/mol_wt["Al2O3"]*2
    Mg_mol = df["MgO"]/mol_wt["MgO"]
    Fe2_mol = df["FeO"]/mol_wt["FeO"]
    Fe3_mol = df["Fe2O3"]/mol_wt["Fe2O3"]*2

    df["Cr#"] = Cr_mol / (Cr_mol + Al_mol)
    df["Mg#"] = Mg_mol / (Mg_mol + Fe2_mol)
    df["Fe*"] = Fe3_mol / (Fe3_mol + Cr_mol + Al_mol)
    
   

    return df


def to_numeric_df(df):
    return df.apply(pd.to_numeric, errors="coerce")



# ========= 组内多数票 + 平均概率 =========
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

# -------------------- 上传文件并处理 --------------------
uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file (please replace your FeO with FeOT, If you did not measure FeO and Fe2O3 separately).", type=["xlsx", "csv"]
)

if uploaded_file is not None:
    try:
        df_uploaded = (pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv")
                       else pd.read_excel(uploaded_file))
        df_uploaded = preprocess_uploaded_data(df_uploaded)

        # 对齐特征列
        df_input = df_uploaded.copy()
        for col in feature_list:
            if col not in df_input.columns: df_input[col] = np.nan
        df_input = to_numeric_df(df_input[feature_list])
        # ✅ 严格对齐列顺序 + 填充缺失值（每个 Level 用自己的 imputer）
        df_input_L1 = _apply_imputer_strict(df_input, imp_L1, "Level1", fallback_cols=feature_list)
        df_input_L2 = _apply_imputer_strict(df_input, imp_L2, "Level2", fallback_cols=feature_list)


        N = len(df_input)

        # ========= Level 1 =========
      
        prob1 = model_lvl1.predict_proba(df_input_L1)
        classes1 = model_lvl1.classes_.astype(str)

        # ===== RAW 概率（不校准）=====
        prob1_use = prob1

        pred1_idx = np.argmax(prob1_use, axis=1)
        pred1_label = classes1[pred1_idx]
        p1max = prob1_use[np.arange(N), pred1_idx]
        # ========= Level 2（仅 Extraterrestrial）=========
        _pred1_norm = pd.Series(pred1_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_lvl2 = (_pred1_norm == "extraterrestrial").to_numpy()

        prob2_raw = np.full((N, len(model_lvl2.classes_)), np.nan)
        pred2_label = np.full(N, "", dtype=object)
        p2max = np.full(N, np.nan)
        p2unk = np.full(N, np.nan)
        classes2 = model_lvl2.classes_.astype(str)

        if mask_lvl2.any():
            pr2 = model_lvl2.predict_proba(df_input_L2[mask_lvl2])

            # ===== RAW 概率（不校准）=====
            pr2_use = pr2

            if thr_L2 is not None:
                pred2_masked, p2max_masked = predict_with_classwise_thresholds(
                    proba_cal=pr2_use, classes=classes2, thr_dict=thr_L2,
                    unknown_label=ABSTAIN_LABEL
                )
            else:
                pred2_masked, p2max_masked = apply_threshold(pr2_use, classes2, 0.5)

            prob2_raw[mask_lvl2] = pr2_use
            pred2_label[mask_lvl2] = pred2_masked
            p2max[mask_lvl2] = p2max_masked
            p2unk[mask_lvl2] = 1.0 - p2max_masked

            # Level2：把 RAW 概率铺回 N 行（其余行填 0）
            prob2_full = np.zeros_like(prob2_raw, dtype=float)
            prob2_full[mask_lvl2] = pr2_use   
        else:
            prob2_full = np.zeros_like(prob2_raw, dtype=float)

        prob2 = np.nan_to_num(prob2_raw, nan=0.0)
        empty2 = (pd.Series(pred2_label, dtype="object") == "")
        if empty2.any():
            pred2_label[empty2.values] = ABSTAIN_LABEL
            p2unk[empty2.values] = 1.0

        # -------------------- 结果表 --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "Index", df_display.index + 1)
        df_display.insert(1, "Level1_Pred", display_level1_array(pred1_label))
        df_display.insert(2, "Level2_Pred", display_level2_array(pred2_label))
        for i, c in enumerate(classes1):
            df_display[f"P_Level1_{display_level1_label(c)}"] = np.round(prob1_use[:, i].astype(float), 3)

        for i, c in enumerate(classes2):
            df_display[f"P_Level2_{display_level2_label(c)}"] = np.round(prob2_full[:, i].astype(float), 3)


      
      

        st.subheader("🧾 Predictions")
        render_big_scroll_table(df_display, height=430, font_px=21)

        # -------------------- 组内多数票 + 均值概率 --------------------
        l1_label, l1_share, l1_mean = level_group_stats(
            labels=pred1_label, classes=classes1, prob_by_class=prob1_use,
            p_max=p1max, p_unknown=None, fill_unknown_for_empty=False
        )

        l2_label, l2_share, l2_mean = level_group_stats(
            labels=pred2_label, classes=classes2, prob_by_class=prob2_full,
            p_max=p2max, p_unknown=p2unk, fill_unknown_for_empty=True
        )

        df_display["L1_TopShare"]    = l1_share
        df_display["L1_TopMeanProb"] = round(l1_mean, 3)
        df_display["L2_TopShare"]    = l2_share
        df_display["L2_TopMeanProb"] = round(l2_mean, 3)

        # -------------------- SHAP：tabs 横向滚动 + 两列并排 --------------------
        st.subheader("📈 SHAP Interpretability")
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"]{
            overflow-x:auto!important;overflow-y:hidden;white-space:nowrap;
            scrollbar-width:thin;-ms-overflow-style:auto;
        }
        .stTabs [data-baseweb="tab"]{white-space:nowrap;padding: 10px 16px;margin:0 3px;font-size:23px!important;}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{ height:8px; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb{ background:rgba(0,0,0,.25); border-radius:8px; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track{ background:rgba(0,0,0,.06); border-radius:8px; }
        .stRadio label {font-size:21px!important;}
        .stRadio [role="radiogroup"] label p {font-size:21px!important;}
        div[data-testid="stMarkdownContainer"] h4 {
            font-size:26px!important;
            margin-bottom:0.5rem!important;
        }
        </style>
        """, unsafe_allow_html=True)

        TOP_K = 13
        st.markdown("<div style='font-size:21px;font-weight:500;margin-bottom:6px;'>Per-class SHAP view</div>", unsafe_allow_html=True)
        chart_kind = st.radio(
            "Per-class SHAP view",
            ["Bar (mean |SHAP|)", "Beeswarm"],
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )

        def _safe_class_names(m):
            try:
                return [str(x) for x in list(getattr(m, "classes_", []))]
            except Exception:
                return []

        def _show_shap_fig_compact(fig):
            # 新版 Streamlit 用 width="content"，旧版则回退到 use_container_width=False。
            # 这样 Matplotlib 图保持自己的尺寸，不再自动铺满整列。
            try:
                st.pyplot(fig, width="content")
            except TypeError:
                st.pyplot(fig, use_container_width=False)

        def _bar_per_class(shap_vals_1class, X, title, top_k=TOP_K):
            mean_abs = np.mean(np.abs(shap_vals_1class), axis=0).reshape(-1)
            order = np.argsort(mean_abs); k = min(top_k, len(order))
            sel = order[-k:]
            feats = np.array(X.columns)[sel]
            vals  = mean_abs[sel]

            # 紧凑版：保留 13 个特征，但不让图占满整个网页。
            fig, ax = plt.subplots(figsize=(3.2*chart_scale, 2.5*chart_scale))
            ax.barh(np.arange(len(vals)), vals)
            ax.set_yticks(np.arange(len(vals)))
            ax.set_yticklabels(feats, fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("mean |SHAP|", fontsize=8)
            ax.set_title(title, fontsize=10, pad=6)
            fig.tight_layout(pad=0.9)
            _show_shap_fig_compact(fig)
            plt.close(fig)

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
            class_names_internal = _safe_class_names(model)
            sv_list = _sv_to_list_per_class(raw_sv, X, class_names_internal)
            if not class_names_internal or len(class_names_internal) != len(sv_list):
                class_names_internal = [f"class {i}" for i in range(len(sv_list))]
                if len(sv_list) == 2:
                    class_names_internal = ["negative", "positive"]

            # 仅替换显示名称；SHAP 数组顺序仍与模型 classes_ 完全一致
            if level_name == "Level2":
                class_names = [display_level2_label(x) for x in class_names_internal]
            elif level_name == "Level1":
                class_names = [display_level1_label(x) for x in class_names_internal]
            else:
                class_names = class_names_internal
            tabs = st.tabs(class_names)
            for tab, cname, arr in zip(tabs, class_names, sv_list):
                with tab:
                    if chart_kind.startswith("Bar"):
                        _bar_per_class(arr, X, title=f"{level_name} · {cname}", top_k=TOP_K)
                    else:
                        shap.summary_plot(arr, X, max_display=TOP_K, show=False)
                        fig = plt.gcf()
                        fig.set_size_inches(3.6*chart_scale, 2.8*chart_scale, forward=True)
                        ax = plt.gca()
                        ax.tick_params(axis="both", labelsize=8)
                        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
                        ax.set_ylabel(ax.get_ylabel(), fontsize=8)
                        plt.title(f"{level_name} · {cname}", fontsize=10, pad=6)
                        # SHAP may create a colorbar as a second axes; enlarge its text too.
                        if len(fig.axes) > 1:
                            for extra_ax in fig.axes[1:]:
                                extra_ax.tick_params(labelsize=8)
                                extra_ax.yaxis.label.set_size(8)
                        plt.tight_layout(pad=0.9)
                        _show_shap_fig_compact(fig)
                        plt.close(fig)

        # 两侧留白 + 中间留白，不让两张图把整行塞满。
        shap_layout = st.columns([1.45, 2.15, 1.8, 2.15, 1.45], gap="small")
        cols_shap = [shap_layout[1], shap_layout[3]]
        X_map = {"Level1": df_input_L1, "Level2": df_input_L2}
        for col, (mdl, nm) in zip(cols_shap, [(model_lvl1, "Level1"), (model_lvl2, "Level2")]):
            with col:
                st.markdown(f"#### 🔍 {nm} (per class)")
                _render_per_class(mdl, nm, X_map[nm])

        # >>> NEW: 预计算 summary（Level1 / Level2）
        # =======================================================================
        def _vc_df_early(labels: np.ndarray) -> pd.DataFrame:
            s = pd.Series(labels, dtype="object").fillna(ABSTAIN_LABEL).replace("", ABSTAIN_LABEL)
            vc = s.value_counts(dropna=False)
            df = vc.rename_axis("Class").reset_index(name="count")
            df["share"] = (df["count"] / float(len(s) if len(s) else 1)).round(3)
            return df[["Class", "count", "share"]]

        # L1 / L2
        df_l1 = _vc_df_early(display_level1_array(pred1_label)).sort_values(["count","Class"], ascending=[False,True], ignore_index=True)
        df_l2 = _vc_df_early(display_level2_array(pred2_label)).sort_values(["count","Class"], ascending=[False,True], ignore_index=True)

        # ===================== 📋 Classification summary (tables)  =====================
        st.subheader("📋 Classification summary (tables)")

        def _make_summary_from_labels(labels, total_n=None) -> pd.DataFrame:
            if labels is None:
                return pd.DataFrame(columns=["Class", "Count", "Share"])
            s = pd.Series(labels, dtype="object").fillna(ABSTAIN_LABEL).replace("", ABSTAIN_LABEL)
            if len(s) == 0:
                return pd.DataFrame(columns=["Class", "Count", "Share"])
            vc = s.value_counts(dropna=False)
            df = vc.rename_axis("Class").reset_index(name="Count")
            denom = float(total_n if (total_n is not None and total_n > 0) else len(s))
            df["Share"] = (df["Count"] / denom).round(3)
            return df[["Class", "Count", "Share"]]

        # 计算 Level2 的有效子集
        pred1_norm = pd.Series(pred1_label, dtype="object").astype("string").str.strip().str.lower().fillna("")
        mask_L2  = (pred1_norm == "extraterrestrial").to_numpy()
        N_L2     = int(mask_L2.sum())

        df_l1_tbl = _make_summary_from_labels(display_level1_array(pred1_label))
        df_l2_tbl = _make_summary_from_labels(display_level2_array(pred2_label[mask_L2]), total_n=N_L2)

        df_l1_tbl.insert(0, "Level", "Level1")
        df_l2_tbl.insert(0, "Level", "Level2")

        cols_tbl = st.columns(2, gap="large")

        with cols_tbl[0]:
            if df_l1_tbl.empty:
                st.info("No data")
            else:
                render_big_scroll_table(df_l1_tbl, height=260, font_px=21)

        with cols_tbl[1]:
            if df_l2_tbl.empty:
                st.info("No Level2 (Extraterrestrial only) data")
            else:
                render_big_scroll_table(df_l2_tbl, height=260, font_px=21)

        # ===================== 🪐 Class share (pie)  =====================
        st.subheader("🪐 Class share (pie)")

        def _vc_df(labels: np.ndarray, total_n: int | None = None) -> pd.DataFrame:
            s = pd.Series(labels, dtype="object").fillna(ABSTAIN_LABEL).replace("", ABSTAIN_LABEL)
            vc = s.value_counts(dropna=False)
            df = vc.rename_axis("Class").reset_index(name="count")
            denom = float(total_n if (total_n is not None and total_n > 0)
                        else (len(s) if len(s) else 1))
            df["share"] = (df["count"] / denom).round(3)
            return df[["Class", "count", "share"]]

        df_pie_l1 = _vc_df(display_level1_array(pred1_label))
        df_pie_l2 = _vc_df(display_level2_array(pred2_label[mask_L2]), total_n=N_L2) if N_L2 > 0 else pd.DataFrame(columns=["Class","count","share"])

        def _fmt_frac(sh: float) -> str:
            if sh >= 0.10:
                return f"{sh:.0%}"
            elif sh >= 0.01:
                return f"{sh:.1%}"
            elif sh >= 0.001:
                return f"{sh:.2%}"
            else:
                return f"{sh:.3%}"

        def _pie_full(col, df: pd.DataFrame, title: str, total_n: int,
                    small_cut: float = 0.06, tiny_cut: float = 0.02):
            cnt_sum = 0
            if df is not None and not df.empty:
                cnt_sum = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).sum()

            with col:
                if cnt_sum == 0 or not total_n:
                    st.info("No data")
                    return

                df_in = df.sort_values(["count", "Class"], ascending=[False, True]).reset_index(drop=True)

                def _collapse_others(df_in: pd.DataFrame, keep_top=8, tiny=0.02):
                    if len(df_in) <= keep_top:
                        out = df_in.copy()
                    else:
                        frac = pd.to_numeric(df_in["count"], errors="coerce").fillna(0) / float(total_n)
                        head = df_in.loc[frac >= tiny].head(keep_top-1)
                        tail = pd.concat([df_in.loc[frac < tiny], df_in.loc[frac >= tiny].iloc[max(keep_top-1,0):]])
                        if len(tail) > 0:
                            others = pd.DataFrame([{
                                "Class": "Others",
                                "count": int(pd.to_numeric(tail["count"], errors="coerce").fillna(0).sum()),
                                "share": float(pd.to_numeric(tail["count"], errors="coerce").fillna(0).sum())/float(total_n)
                            }])
                            out = pd.concat([head, others], ignore_index=True)
                        else:
                            out = head
                    s = float(pd.to_numeric(out["count"], errors="coerce").fillna(0).sum()) or 1.0
                    out["share"] = pd.to_numeric(out["count"], errors="coerce").fillna(0)/s
                    return out

                df_plot = _collapse_others(df_in, keep_top=8, tiny=tiny_cut)
                labels = df_plot["Class"].astype(str).tolist()
                sizes  = pd.to_numeric(df_plot["count"], errors="coerce").fillna(0).astype(int).to_numpy()
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

                def _autopct(pct):
                    return f"{pct:.0f}%" if (pct/100.0) >= small_cut else ""

                fig, ax = plt.subplots(figsize=(3.6*chart_scale, 2.8*chart_scale))
                wedges, texts, autotexts = ax.pie(
                    sizes, startangle=110, counterclock=False,
                    colors=colors, labels=None,
                    autopct=_autopct, pctdistance=0.72,
                    labeldistance=1.10,
                    wedgeprops=dict(linewidth=0.9, edgecolor="white"),
                    textprops=dict(fontsize=8)
                )

                legend_labels = [f"{_short_chart_label(lab)}, {_fmt_frac(sh)}" for lab, sh in zip(df_in["Class"], df_in["share"])]
                ax.legend(
                    wedges, legend_labels, title="Class",
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=8,
                    title_fontsize=8
                )
                ax.axis("equal")
                ax.set_title(title, fontsize=10, pad=6)

                _show_shap_fig_compact(fig)
                st.download_button(
                    "⬇️ Download PNG",
                    _save_fig_as_png_bytes(fig, dpi=int(220*chart_scale)),
                    file_name=f"{title.replace(' · ','_').replace(' ','_')}.png",
                    mime="image/png"
                )
                plt.close(fig)

        pie_layout = st.columns([1.45, 2.15, 1.8, 2.15, 1.45], gap="small")
        cols_pie = [pie_layout[1], pie_layout[3]]
        _pie_full(cols_pie[0], df_pie_l1, "Level1 · class share", total_n=len(pred1_label))
        _pie_full(cols_pie[1], df_pie_l2, "Level2 · class share (Extraterrestrial only)", total_n=(N_L2 if N_L2 > 0 else 1))

        # ===================== ☄️ Class frequency (bars)  =====================
        st.subheader("☄️ Class frequency (bars)")

        def _bar_from_df(col, df: pd.DataFrame, title: str, total_n: int):
            with col:
                if df.empty or int(df["count"].sum() or 0) == 0:
                    st.info("No data"); return
                fig, ax = plt.subplots(figsize=(3.2*chart_scale, 2.5*chart_scale))
                x = [_short_chart_label(v) for v in df["Class"].astype(str).tolist()]
                y = df["count"].astype(int).tolist()
                ax.bar(range(len(x)), y, edgecolor="black", color=[PALETTE[i % len(PALETTE)] for i in range(len(x))])
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels(x, rotation=28, ha="right", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_title(title, fontsize=10)

                ymax = max(max(y), 1)
                ax.set_ylim(0, ymax * 1.18)
                for i, yi in enumerate(y):
                    ax.text(i, yi + ymax * 0.02, f"{yi}/{total_n}", ha="center", va="bottom", fontsize=8)

                plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.26)
                _show_shap_fig_compact(fig)
                st.download_button(
                    "⬇️ Download PNG",
                    _save_fig_as_png_bytes(fig, dpi=int(220*chart_scale)),
                    file_name=f"{title.replace(' · ','_').replace(' ','_')}.png",
                    mime="image/png"
                )
                plt.close(fig)

        bar_layout = st.columns([1.45, 2.15, 1.8, 2.15, 1.45], gap="small")
        cols_bar = [bar_layout[1], bar_layout[3]]
        _bar_from_df(cols_bar[0], df_pie_l1.sort_values(["count","Class"], ascending=[False,True]), "Level1 · frequency", total_n=len(pred1_label))
        _bar_from_df(cols_bar[1], df_pie_l2.sort_values(["count","Class"], ascending=[False,True]), "Level2 · frequency (Extraterrestrial only)", total_n=N_L2 if N_L2>0 else 1)

        # -------------------- ✅ 样品一致性 + 组结果 --------------------
        st.subheader("🧪 Specimen Confirmation & Group Result")
        same_specimen = st.checkbox("I confirm all uploaded rows originate from the same physical specimen.")
        if same_specimen:
            depth = {"Level1": 1, "Level2": 2}
            cands = [
                ("Level1", {"label": l1_label, "share": int(l1_share.split('/')[0]) / N, "prob": l1_mean,
                            "agree": int(l1_share.split('/')[0]), "total": N}),
                ("Level2", {"label": l2_label, "share": int(l2_share.split('/')[0]) / N, "prob": l2_mean,
                            "agree": int(l2_share.split('/')[0]), "total": N}),
            ]
            final_level, final = sorted(cands, key=lambda t: (t[1]["share"], t[1]["prob"], depth[t[0]]), reverse=True)[0]
            final_label_display = (
                display_level2_label(final["label"])
                if final_level == "Level2"
                else display_level1_label(final["label"])
            )
            st.success(
                f"Final group result → **{final_level}: {final_label_display}**  |  "
                f"Probability (mean for this class): **{final['prob']:.3f}**  |  "
                f"Share: **{final['agree']}/{final['total']} ({final['share']:.0%})**"
            )
            rows = [
                {"Level": "Level1", "Top class": display_level1_label(l1_label), "Share": l1_share, "Mean prob": round(l1_mean, 3)},
                {"Level": "Level2", "Top class": display_level2_label(l2_label), "Share": l2_share, "Mean prob": round(l2_mean, 3)},
            ]
            render_big_scroll_table(pd.DataFrame(rows), height=220, font_px=21)

        # -------------------- 训练池（在直方图后，下载前） --------------------
        st.subheader("🧩 Add Predictions to Training Pool?")
        if st.checkbox("✅ Confirm to append these samples to the training pool for future retraining"):
            df_save = df_input.copy()
            df_save["Level1"] = pred1_label
            df_save["Level2"] = pred2_label

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

        # -------------------- 结果下载（Prediction + Summary） --------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # 预测页（包含 OOD 列）
            _round_float_columns(df_display, 3).to_excel(writer, index=False, sheet_name='Prediction')

            # 导出 Level1 / Level2 汇总
            df_l1_export = df_l1.copy(); df_l1_export.insert(0, "Level", "Level1")
            df_l2_export = df_l2.copy(); df_l2_export.insert(0, "Level", "Level2")
            _round_float_columns(df_l1_export, 3).to_excel(writer, index=False, sheet_name='Summary_L1')
            _round_float_columns(df_l2_export, 3).to_excel(writer, index=False, sheet_name='Summary_L2')

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
