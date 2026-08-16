import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os, joblib, requests, base64, re
from io import BytesIO
from itertools import chain
from matplotlib.patches import Patch

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="Chromite Provenance Classifier", layout="wide")
def render_hero_banner():
    hero_path = "chromite_space_banner.png" if os.path.exists("chromite_space_banner.png") else None

    if hero_path:
        with open(hero_path, "rb") as f:
            hero_b64 = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <div class="hero-banner-wrap"
                 style="background-image: url('data:image/png;base64,{hero_b64}');">
                <div class="hero-copy">
                    <div class="hero-banner-title">Chromite Provenance Classifier</div>
                    <div class="hero-banner-subtitle">
                        Machine-learning classification of chromite compositions for terrestrial–extraterrestrial provenance and extraterrestrial subclass attribution.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Hero background image not found: chromite_space_banner.png")

# -------------------- 网页字体 --------------------
st.markdown("""
<style>

/* ---------- hero banner ---------- */
.hero-banner-wrap {
    width: 100%;
    height: clamp(210px, 13vw, 300px);
    margin: 6px 0 14px 0;
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(12, 24, 48, 0.22);
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    display: flex;
    align-items: center;
    overflow: hidden;
}

.hero-copy {
    width: 100%;
    padding: 0 3.2vw;
    box-sizing: border-box;
    color: #ffffff;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.62);
}

.hero-banner-title {
    margin: 0 0 12px 0;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(30px, 2.7vw, 58px);
    font-weight: 700;
    line-height: 1.05;
    color: #ffffff;
    white-space: nowrap;
}

.hero-banner-subtitle {
    margin: 0;
    font-size: clamp(13px, 1.05vw, 21px);
    font-weight: 500;
    line-height: 1.25;
    color: rgba(255, 255, 255, 0.94);
    white-space: nowrap;
}

/* Keep both hero lines on one row while scaling down on narrower screens */
@media (max-width: 1200px) {
    .hero-banner-title {
        font-size: 34px;
    }
    .hero-banner-subtitle {
        font-size: 13px;
    }
}

@media (max-width: 900px) {
    .hero-banner-title {
        font-size: 26px;
    }
    .hero-banner-subtitle {
        font-size: 10px;
    }
}

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
    font-size: 24px !important;
}
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] > label p {
    font-size: 20px !important;
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


/* ---------- research UI refinements ---------- */
.research-badge {
    display: inline-block;
    padding: 11px 17px;
    border: 1px solid #dbe3ec;
    border-radius: 999px;
    background: #f7f9fc;
    color: #354052;
    font-size: 15px;
    font-weight: 600;
}
.result-card {
    border: 1px solid #e1e7ef;
    border-radius: 12px;
    background: #fbfcfe;
    padding: 16px 18px;
    min-height: 118px;
}
.result-card .card-label {
    color: #6a7380;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 7px;
}
.result-card .card-value {
    color: #1f2937;
    font-size: 22px;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 7px;
}
.result-card .card-meta {
    color: #697386;
    font-size: 15px;
    line-height: 1.35;
}
/* ---------- highlighted expandable feature panels ---------- */
div[data-testid="stExpander"] {
    border: 1px solid #bdd5ef !important;
    border-radius: 12px !important;
    background: #f8fbff !important;
    overflow: hidden !important;
    margin-top: 8px !important;
    margin-bottom: 12px !important;
}

/* Streamlit 1.60 expander header: keep the label genuinely centered */
div[data-testid="stExpander"] details > summary {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 60px !important;
    padding: 0 68px !important;
    box-sizing: border-box !important;
    background: #eaf4ff !important;
    border-radius: 11px !important;
    list-style: none !important;
    cursor: pointer !important;
    transition: background-color 0.18s ease, border-color 0.18s ease;
}

/* remove the browser's own details marker */
div[data-testid="stExpander"] details > summary::-webkit-details-marker {
    display: none !important;
}

div[data-testid="stExpander"] details > summary:hover {
    background: #dceeff !important;
}

/* In current Streamlit the expander label is carried by a span, not the old p wrapper. */
div[data-testid="stExpander"] details > summary > span {
    flex: 1 1 auto !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    text-align: center !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    line-height: 1.35 !important;
    color: #245b8f !important;
}

/* also cover any nested markdown/text wrapper Streamlit inserts */
div[data-testid="stExpander"] details > summary > span *,
div[data-testid="stExpander"] details > summary p {
    margin: 0 !important;
    padding: 0 !important;
    text-align: center !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    line-height: 1.35 !important;
    color: #245b8f !important;
}

/* Hide Streamlit's tiny built-in chevron and draw a larger, cleaner one on the right. */
div[data-testid="stExpander"] details > summary > svg {
    display: none !important;
}

div[data-testid="stExpander"] details > summary::after {
    content: "▾";
    position: absolute !important;
    right: 22px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    transform-origin: center !important;
    font-family: Arial, sans-serif !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    color: #245b8f !important;
    pointer-events: none !important;
    transition: transform 0.18s ease !important;
}

div[data-testid="stExpander"] details[open] > summary::after {
    transform: translateY(-50%) rotate(180deg) !important;
}

/* keep the open-panel body clean */
div[data-testid="stExpander"] details[open] > summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom: 1px solid #d4e4f5 !important;
}

/* ---------- file uploader: keep native controls, enlarge them, and preserve the remove-file button ---------- */
div[data-testid="stFileUploader"] {
    width: 100% !important;
}

/* Main upload/drop-zone container */
div[data-testid="stFileUploader"] section {
    min-height: 76px !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
}

/* Upload/Browse button: restore the larger, easier-to-read appearance */
div[data-testid="stFileUploader"] button {
    min-height: 44px !important;
    padding: 8px 15px !important;
}

div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button p,
div[data-testid="stFileUploader"] button span {
    font-size: 20px !important;
    line-height: 1.2 !important;
}

/* File-type / size hint */
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] section > div > span {
    font-size: 19px !important;
}

/* IMPORTANT: do not hide uploader SVGs. The X/remove-file control is an SVG button. */
div[data-testid="stFileUploader"] button svg {
    display: block !important;
    width: 20px !important;
    height: 20px !important;
}
.footer-note {
    color: #8a93a0;
    font-size: 13px;
    text-align: center;
    padding: 18px 0 8px 0;
}


/* 强制放大 SHAP 类别 tabs */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [role="tab"] p,
.stTabs [role="tab"] span {
    font-size: 27px !important;
}



</style>
""", unsafe_allow_html=True)

render_hero_banner()

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
    "mars": "Mars",
    "martian": "Mars",
    "martian meteorite": "Mars",
    "martian meteorites": "Mars",
    "brachinite": "brachinites",
    "brachinites": "brachinites",
    "lunar": "Lunar",
    "lunar meteorite": "Lunar",
    "lunar meteorites": "Lunar",
    "low-ti lunar": "low-Ti lunar",
    "low-ti lunar sample": "low-Ti lunar",
    "low-ti lunar samples": "low-Ti lunar",
    "high-ti lunar": "high-Ti lunar",
    "high-ti lunar sample": "high-Ti lunar",
    "high-ti lunar samples": "high-Ti lunar",
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


# -------------------- 特征显示映射（用于图中下标） --------------------
FEATURE_DISPLAY_MAP = {
    "MgO": "MgO",
    "Al2O3": "Al₂O₃",
    "TiO2": "TiO₂",
    "V2O3": "V₂O₃",
    "Cr2O3": "Cr₂O₃",
    "MnO": "MnO",
    "FeOT": r"FeO$_T$",
    "FeO": "FeO",
    "Fe2O3": "Fe₂O₃",
    "FeOre": "FeO",
    "Fe2O3re": "Fe₂O₃",
    "FeO_total": "FeO total",
    "SiO2": "SiO₂",
    "ZnO": "ZnO",
    "Cr#": "Cr#",
    "Mg#": "Mg#",
    "Fe*": "Fe*",
    "Fe2_frac": "Fe²⁺ frac",
    "Fe3_frac": "Fe³⁺ frac",
    "TAC": "TAC",
    "Total": "Total",
    "Unnamed: 0": "Unnamed: 0",
}

def display_feature_label(x: str) -> str:
    return FEATURE_DISPLAY_MAP.get(str(x), str(x))

def display_feature_labels(seq):
    return [display_feature_label(x) for x in seq]

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
    st.caption("Build: auto-upload-polished-v42")

    
    @st.cache_resource
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
@st.cache_resource
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

# -------------------- 输入模板下载 --------------------
template_path = "chromite_input_template.xlsx"
if os.path.exists(template_path):
    with open(template_path, "rb") as f:
        template_bytes = f.read()

    # 模板下载按钮：真正贴右对齐（只影响这个按钮，不影响页面内其他下载按钮）
    st.markdown(
        """
        <style>
        .st-key-download_input_template {
            width: 100% !important;
            display: flex !important;
            justify-content: flex-end !important;
        }
        .st-key-download_input_template > div {
            width: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download input template (Excel)",
        data=template_bytes,
        file_name="chromite_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_input_template"
    )
else:
    st.warning(
        "Input template file not found. Please add chromite_input_template.xlsx "
        "to the same repository folder as streamlit_app.py."
    )

# -------------------- 上传文件并处理 --------------------
uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file (please replace your FeO with FeOT, If you did not measure FeO and Fe2O3 separately).",
    type=["xlsx", "csv"],
    key="chromite_data_uploader"
)

if uploaded_file is not None:
    # Process immediately after a file is selected; no extra "Run classification" click is required.
    file_size_kb = float(getattr(uploaded_file, "size", 0)) / 1024.0
    st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size_kb:.1f} KB)")

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

        # -------------------- 结果表 + batch overview --------------------
        df_display = df_uploaded.copy().reset_index(drop=True)
        df_display.insert(0, "Index", df_display.index + 1)
        df_display.insert(1, "Level1_Pred", display_level1_array(pred1_label))
        df_display.insert(2, "Level2_Pred", display_level2_array(pred2_label))
        for i, c in enumerate(classes1):
            df_display[f"P_Level1_{display_level1_label(c)}"] = np.round(prob1_use[:, i].astype(float), 3)

        for i, c in enumerate(classes2):
            df_display[f"P_Level2_{display_level2_label(c)}"] = np.round(prob2_full[:, i].astype(float), 3)

        # 组内多数票 + 均值概率（先计算，让主要结果出现在大表之前）
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

        # -------------------- Compact prediction table --------------------
        st.subheader("🧾 Prediction Results")
        df_preview = pd.DataFrame({
            "Index": df_display["Index"],
            "Level 1 prediction": df_display["Level1_Pred"],
            "Level 1 predicted probability": np.round(p1max, 3),
            "Level 2 prediction": df_display["Level2_Pred"],
            "Level 2 predicted probability": np.round(p2max, 3),
        })
        render_big_scroll_table(df_preview, height=320, font_px=21)

        with st.expander("Show full analytical details and class probabilities", expanded=False):
            render_big_scroll_table(df_display, height=430, font_px=19)

        # -------------------- SHAP：tabs 横向滚动 + 两列并排 --------------------
        st.subheader("📈 SHAP Interpretability")
        st.caption("Feature contributions to class predictions. Switch between global importance bars and beeswarm views.")
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"]{
            overflow-x:auto!important;overflow-y:hidden;white-space:nowrap;
            scrollbar-width:thin;-ms-overflow-style:auto;
        }
        .stTabs [data-baseweb="tab"]{white-space:nowrap;padding: 10px 16px;margin:0 3px;font-size:27px!important;}
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


        # ===== 统计图统一画布 / 字体 / 网页显示尺寸 =====
        STATS_FIGSIZE = (7.2, 4.9)   # 四张图完全相同，稍微放宽以容纳 legend
        STATS_DPI = 120
        STATS_DISPLAY_WIDTH = 680    # 网页上四张图完全相同宽度
        STATS_FONT = 15
        STATS_TITLE_FONT = 17

        def _stats_png_bytes(fig):
            buf = BytesIO()
            # 不使用 bbox_inches="tight"，避免不同内容导致最终图片尺寸变化
            fig.savefig(
                buf,
                format="png",
                dpi=STATS_DPI,
                bbox_inches=None,
                facecolor="white"
            )
            buf.seek(0)
            return buf.getvalue()

        def _show_fixed_stats_fig(fig, title):
            png = _stats_png_bytes(fig)
            st.image(png, width=STATS_DISPLAY_WIDTH)
            st.download_button(
                "⬇️ Download PNG",
                png,
                file_name=f"{title.replace(' · ','_').replace(' ','_')}.png",
                mime="image/png"
            )

        def _bar_per_class(shap_vals_1class, X, title, top_k=TOP_K):
            mean_abs = np.mean(np.abs(shap_vals_1class), axis=0).reshape(-1)
            order = np.argsort(mean_abs); k = min(top_k, len(order))
            sel = order[-k:]
            feats = np.array(display_feature_labels(X.columns))[sel]
            vals  = mean_abs[sel]

            # 紧凑版：保留 13 个特征，但不让图占满整个网页。
            fig, ax = plt.subplots(figsize=(5.8*chart_scale, 4.3*chart_scale))
            ax.barh(np.arange(len(vals)), vals)
            ax.set_yticks(np.arange(len(vals)))
            ax.set_yticklabels(feats, fontsize=10)
            ax.tick_params(axis="x", labelsize=10)
            ax.set_xlabel("mean |SHAP|", fontsize=10)
            ax.set_title(title, fontsize=12, pad=8)
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
                        X_disp = X.copy()
                        X_disp.columns = display_feature_labels(X.columns)
                        shap.summary_plot(arr, X_disp, max_display=TOP_K, show=False)
                        fig = plt.gcf()
                        fig.set_size_inches(5.8*chart_scale, 4.3*chart_scale, forward=True)
                        ax = plt.gca()
                        ax.tick_params(axis="both", labelsize=10)
                        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                        ax.set_ylabel(ax.get_ylabel(), fontsize=10)
                        plt.title(f"{level_name} · {cname}", fontsize=12, pad=8)
                        # SHAP may create a colorbar as a second axes; enlarge its text too.
                        if len(fig.axes) > 1:
                            for extra_ax in fig.axes[1:]:
                                extra_ax.tick_params(labelsize=10)
                                extra_ax.yaxis.label.set_size(10)
                        plt.tight_layout(pad=0.9)
                        _show_shap_fig_compact(fig)
                        plt.close(fig)

        # 两侧留白 + 中间留白，不让两张图把整行塞满。
        shap_layout = st.columns([1.10, 2.70, 1.10, 2.70, 1.10], gap="small")
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
        st.caption("Distribution of predicted classes across the uploaded analyses.")

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

        # 两张统计表收窄：两侧留白 + 中间留白
        tbl_layout = st.columns([1.25, 2.55, 1.00, 2.55, 1.25], gap="small")
        cols_tbl = [tbl_layout[1], tbl_layout[3]]

        with cols_tbl[0]:
            if df_l1_tbl.empty:
                st.info("No data")
            else:
                render_big_scroll_table(df_l1_tbl, height=145, font_px=21)

        with cols_tbl[1]:
            if df_l2_tbl.empty:
                st.info("No Level2 (Extraterrestrial only) data")
            else:
                # Level 2 usually contains more classes: show up to five data rows by default.
                # Additional classes remain accessible with the table's internal scrollbar.
                visible_rows_l2 = min(5, len(df_l2_tbl))
                l2_table_height = 48 + visible_rows_l2 * 46
                render_big_scroll_table(df_l2_tbl, height=l2_table_height, font_px=21)

        # ===================== 🪐 Classification distribution figures =====================
        st.subheader("🪐 Classification distribution figures")
        st.caption("Each panel combines a class-share pie chart, a frequency bar chart, and a color legend.")

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

        def _combined_distribution_figure(col, df: pd.DataFrame, title: str, total_n: int,
                                          small_cut: float = 0.06, tiny_cut: float = 0.02):
            with col:
                cnt_sum = 0
                if df is not None and not df.empty:
                    cnt_sum = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).sum()

                if cnt_sum == 0 or not total_n:
                    st.info("No data")
                    return

                df_in = df.sort_values(["count", "Class"], ascending=[False, True]).reset_index(drop=True)

                def _collapse_others(df_in: pd.DataFrame, keep_top=8, tiny=0.02):
                    if len(df_in) <= keep_top:
                        out = df_in.copy()
                    else:
                        frac = pd.to_numeric(df_in["count"], errors="coerce").fillna(0) / float(total_n)
                        head = df_in.loc[frac >= tiny].head(keep_top - 1)
                        tail = pd.concat([
                            df_in.loc[frac < tiny],
                            df_in.loc[frac >= tiny].iloc[max(keep_top - 1, 0):]
                        ])
                        if len(tail) > 0:
                            others_count = int(pd.to_numeric(tail["count"], errors="coerce").fillna(0).sum())
                            others = pd.DataFrame([{
                                "Class": "Others",
                                "count": others_count,
                                "share": float(others_count) / float(total_n)
                            }])
                            out = pd.concat([head, others], ignore_index=True)
                        else:
                            out = head
                    s = float(pd.to_numeric(out["count"], errors="coerce").fillna(0).sum()) or 1.0
                    out["share"] = pd.to_numeric(out["count"], errors="coerce").fillna(0) / s
                    return out

                df_plot = _collapse_others(df_in, keep_top=8, tiny=tiny_cut)
                labels = df_plot["Class"].astype(str).tolist()
                counts = pd.to_numeric(df_plot["count"], errors="coerce").fillna(0).astype(int).to_numpy()
                short_labels = [_short_chart_label(x) for x in labels]
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

                # One large panel: pie on the left, frequency bars on the right,
                # color legend spanning the full width underneath.
                fig = plt.figure(figsize=(10.6, 6.6))
                gs = fig.add_gridspec(
                    2, 2,
                    height_ratios=[3.55, 1.45],
                    width_ratios=[0.90, 1.15],
                    hspace=0.32,
                    wspace=0.36
                )
                ax_pie = fig.add_subplot(gs[0, 0])
                ax_bar = fig.add_subplot(gs[0, 1])
                ax_leg = fig.add_subplot(gs[1, :])
                ax_leg.axis("off")

                def _autopct(pct):
                    return f"{pct:.0f}%" if (pct / 100.0) >= small_cut else ""

                ax_pie.pie(
                    counts,
                    startangle=110,
                    counterclock=False,
                    colors=colors,
                    labels=None,
                    autopct=_autopct,
                    pctdistance=0.70,
                    radius=0.88,
                    wedgeprops=dict(linewidth=0.9, edgecolor="white"),
                    textprops=dict(fontsize=14)
                )
                ax_pie.axis("equal")
                ax_pie.set_title("Class share", fontsize=17, pad=8)

                x = np.arange(len(labels))
                ax_bar.bar(x, counts, edgecolor="black", linewidth=0.8, color=colors)
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=12)
                ax_bar.set_ylabel("Count", fontsize=15)
                ax_bar.tick_params(axis="y", labelsize=13)
                ax_bar.set_title("Class frequency", fontsize=17, pad=8)

                ymax = max(max(counts), 1)
                ax_bar.set_ylim(0, ymax * 1.22)
                for i, yi in enumerate(counts):
                    ax_bar.text(
                        i, yi + ymax * 0.025,
                        f"{yi}/{total_n}",
                        ha="center", va="bottom", fontsize=12
                    )

                legend_labels = [
                    f"{_short_chart_label(lab)}: {cnt}/{total_n} ({_fmt_frac(cnt / float(total_n))})"
                    for lab, cnt in zip(labels, counts)
                ]
                if len(labels) <= 3:
                    ncol = 1
                elif len(labels) <= 6:
                    ncol = 2
                else:
                    ncol = 3
                leg = ax_leg.legend(
                    handles=[Patch(facecolor=c, edgecolor="none") for c in colors],
                    labels=legend_labels,
                    title="Color legend",
                    loc="center",
                    bbox_to_anchor=(0.5, 0.42),
                    ncol=ncol,
                    frameon=False,
                    fontsize=12,
                    title_fontsize=14,
                    columnspacing=1.8,
                    handlelength=1.3,
                    handletextpad=0.55,
                    labelspacing=0.85,
                    borderaxespad=0.0
                )
                try:
                    leg._legend_box.align = "left"
                except Exception:
                    pass

                fig.suptitle(title, fontsize=19, y=0.955)
                fig.subplots_adjust(left=0.08, right=0.94, top=0.86, bottom=0.10)

                png = _stats_png_bytes(fig)
                st.image(png, width="stretch")
                st.download_button(
                    "⬇️ Download PNG",
                    png,
                    file_name=f"{title.replace(' · ','_').replace(' ','_')}.png",
                    mime="image/png",
                    key=f"download_distribution_{re.sub(r'[^A-Za-z0-9]+', '_', title)}"
                )
                plt.close(fig)

        combined_layout = st.columns([1, 1], gap="medium")
        _combined_distribution_figure(
            combined_layout[0],
            df_pie_l1,
            "Level 1 classification",
            total_n=len(pred1_label)
        )
        _combined_distribution_figure(
            combined_layout[1],
            df_pie_l2,
            "Level 2 classification (Extraterrestrial only)",
            total_n=(N_L2 if N_L2 > 0 else 1)
        )

        # -------------------- 自愿数据分享（默认折叠） --------------------
        with st.expander(
            "Would you like to share your data with us to help expand the database and improve future model retraining?",
            expanded=False
        ):
            st.caption(
                "Shared data will be used for research database development and future model retraining. "
                "Please provide a contact email so we can follow up if clarification is needed."
            )

            contact_email = st.text_input(
                "Contact email",
                placeholder="name@example.com",
                key="data_share_contact_email"
            ).strip()

            same_specimen = st.checkbox(
                "I confirm all uploaded rows originate from the same physical specimen.",
                key="same_physical_specimen"
            )

            if same_specimen:
                depth = {"Level1": 1, "Level2": 2}
                cands = [
                    ("Level1", {
                        "label": l1_label,
                        "share": int(l1_share.split('/')[0]) / N,
                        "prob": l1_mean,
                        "agree": int(l1_share.split('/')[0]),
                        "total": N
                    }),
                    ("Level2", {
                        "label": l2_label,
                        "share": int(l2_share.split('/')[0]) / N,
                        "prob": l2_mean,
                        "agree": int(l2_share.split('/')[0]),
                        "total": N
                    }),
                ]
                final_level, final = sorted(
                    cands,
                    key=lambda t: (t[1]["share"], t[1]["prob"], depth[t[0]]),
                    reverse=True
                )[0]

                final_label_display = (
                    display_level2_label(final["label"])
                    if final_level == "Level2"
                    else display_level1_label(final["label"])
                )

                st.success(
                    f"Group result → **{final_level}: {final_label_display}**  |  "
                    f"Mean probability: **{final['prob']:.3f}**  |  "
                    f"Share: **{final['agree']}/{final['total']} ({final['share']:.0%})**"
                )

                rows = [
                    {
                        "Level": "Level1",
                        "Top class": display_level1_label(l1_label),
                        "Share": l1_share,
                        "Mean prob": round(l1_mean, 3)
                    },
                    {
                        "Level": "Level2",
                        "Top class": display_level2_label(l2_label),
                        "Share": l2_share,
                        "Mean prob": round(l2_mean, 3)
                    },
                ]
                render_big_scroll_table(pd.DataFrame(rows), height=220, font_px=21)

            share_consent = st.checkbox(
                "I agree to share these uploaded data with the research database for future model retraining.",
                key="share_training_consent"
            )

            # A deliberate submit button prevents accidental / repeated writes on ordinary Streamlit reruns.
            if share_consent:
                email_ok = bool(
                    re.fullmatch(
                        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                        contact_email
                    )
                )

                if not contact_email:
                    st.info("Please enter a contact email before submitting the data.")
                elif not email_ok:
                    st.warning("Please enter a valid email address.")
                else:
                    submit_share = st.button(
                        "Submit data to the research database",
                        type="primary",
                        key="submit_shared_training_data"
                    )

                    if submit_share:
                        df_save = df_input.copy()
                        df_save["Level1"] = pred1_label
                        df_save["Level2"] = pred2_label
                        df_save["ContactEmail"] = contact_email
                        df_save["SamePhysicalSpecimen"] = bool(same_specimen)
                        df_save["SourceFile"] = str(getattr(uploaded_file, "name", "uploaded_file"))

                        local_path = "training_pool.csv"
                        header_needed = not os.path.exists(local_path)
                        df_save.to_csv(
                            local_path,
                            mode="a",
                            header=header_needed,
                            index=False,
                            encoding="utf-8-sig"
                        )
                        st.success("✅ Thank you. Your data have been added to the research training pool.")

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
                                st.info(
                                    "GitHub token is not configured. "
                                    "The submission was saved to the app's local training pool only."
                                )
                            else:
                                with open(local_path, "rb") as f:
                                    content_b64 = base64.b64encode(f.read()).decode("utf-8")

                                url = (
                                    f"https://api.github.com/repos/"
                                    f"{repo_owner}/{repo_name}/contents/{dst_path}"
                                )
                                headers = {
                                    "Authorization": f"token {GITHUB_TOKEN}",
                                    "Accept": "application/vnd.github+json"
                                }

                                r = requests.get(url, headers=headers)
                                sha = r.json().get("sha") if r.status_code == 200 else None

                                payload = {
                                    "message": "update training pool",
                                    "content": content_b64,
                                    "branch": branch
                                }
                                if sha:
                                    payload["sha"] = sha

                                put_resp = requests.put(
                                    url,
                                    headers=headers,
                                    json=payload
                                )

                                if 200 <= put_resp.status_code < 300:
                                    st.success("✅ Shared data synchronized to the research repository.")
                                else:
                                    st.warning(
                                        f"⚠️ Repository sync failed ({put_resp.status_code}): "
                                        f"{put_resp.text[:300]}"
                                    )
                        except Exception as e:
                            st.error(f"❌ Repository sync error: {e}")

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

st.markdown(
    """
    <div class="footer-note">
        Chromite Provenance Classifier · Research-use classification interface
    </div>
    """,
    unsafe_allow_html=True
)
