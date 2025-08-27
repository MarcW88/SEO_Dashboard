# app.py — Vue stratégique (KPI globaux DataForSEO + meilleur thème via tes labels)
# Menus "pills" au-dessus des KPI + multiselect "badges" au-dessus du radar

import os, json, time, unicodedata, re
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Tuple, Optional
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API_BASE = "https://api.dataforseo.com/v3"

# >>> paramètres qui matchent ton test DataForSEO (Belgium FR)
LOCATION_CODE = 2056
LANGUAGE_CODE = "fr"
LIMIT_RK     = 5000

INTENTS = {"Informational", "Navigational", "Transactional", "Branding", "Product-branding"}

st.set_page_config(page_title="SEO Coverage — Strategic KPIs", page_icon="🧭", layout="wide")
st.title("Dashboard")

# -------------------- Credentials & HTTP --------------------
def get_dfs_creds() -> Tuple[str, str]:
    login = os.getenv("DATAFORSEO_LOGIN", "")
    password = os.getenv("DATAFORSEO_PASSWORD", "")
    if not login or not password:
        try:
            login = login or st.secrets.get("DATAFORSEO_LOGIN", "")
            password = password or st.secrets.get("DATAFORSEO_PASSWORD", "")
        except Exception:
            pass
    return login, password

def _auth_headers(login: str, password: str) -> dict:
    import base64
    tok = base64.b64encode(f"{login}:{password}".encode()).decode()
    return {"Authorization": f"Basic {tok}", "Content-Type": "application/json"}

def safe_post(url: str, headers: dict, payload: list, timeout: int = 90) -> dict:
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        try:
            err = r.json()
        except Exception:
            err = str(e)
        st.error(f"API error on {url}: {err}")
        return {}

# -------------------------- Utils ---------------------------
def extract_root_domain(url: str) -> str:
    if not url: return ""
    host = urlparse(url if url.startswith("http") else "https://" + url).netloc.lower().strip()
    return host[4:] if host.startswith("www.") else host

def normalize_kw_basic(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = s.replace("’", "'")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_kw_strict(s: str) -> str:
    nk = normalize_kw_basic(s)
    nk = "".join(ch for ch in unicodedata.normalize("NFKD", nk) if not unicodedata.combining(ch))
    return " ".join(nk.split())

def primary_theme(labels: List[str]) -> str:
    if not isinstance(labels, list) or not labels: return "Unlabeled"
    themes = [l for l in labels if l not in INTENTS]
    return themes[0] if themes else "Unlabeled"

def tok_norm(s: str) -> set:
    s = normalize_kw_strict(s)
    toks = re.findall(r"[a-z0-9]+", s)
    return {t for t in toks if len(t) >= 2}

# ------------------ Charge JSON sémantique -------------------
def load_semantic_json(path: str = "data/semantic_keywords.json") -> dict:
    p = Path(path)
    if not p.exists():
        st.error("Fichier manquant: data/semantic_keywords.json")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"JSON invalide: {e}")
        return {}

raw = load_semantic_json()
if not raw: st.stop()

kw_map = raw.get("keywords", {})
specific_url = raw.get("specific_url", "")
target_domain = extract_root_domain(specific_url)
if not target_domain:
    st.error("`specific_url` absent ou invalide (impossible d'extraire un domaine).")
    st.stop()

labels_df = (
    pd.DataFrame([{"keyword": k.strip(), "labels": v or []} for k, v in kw_map.items()])
      .drop_duplicates(subset=["keyword"])
      .reset_index(drop=True)
)
labels_df["kw_norm2"] = labels_df["keyword"].apply(normalize_kw_strict)

# ======================= UI — PÉRIODE (pills) =======================
st.markdown("""
<style>
.filters-bar {display:flex; gap:.5rem; align-items:center; margin:.25rem 0 .75rem 0;}
.pill {border:1px solid #e5e7eb; border-radius:12px; padding:.35rem .6rem; background:#f8fafc; font-size:0.9rem;}
.pill .label {color:#4b5563; margin-right:.35rem;}
</style>
""", unsafe_allow_html=True)

spacer, wf_col, date_col = st.columns([4,1,4])
with wf_col:
    with_filters = st.toggle("with filters", value=True)
with date_col:
    compare_prev = st.toggle("Comparer période précédente", value=True)
    if with_filters:
        default_start = date.today() - timedelta(days=29)
        date_range = st.date_input(
            "",
            value=(default_start, date.today()),
            label_visibility="collapsed",
        )
    else:
        fixed_start = date.today() - timedelta(days=29)
        date_range = (fixed_start, date.today())
        st.date_input("", value=date_range, disabled=True, label_visibility="collapsed")
    date_note = st.empty()

window = (date_range[1] - date_range[0]).days + 1
start_A, end_d = date_range[0], date_range[1]

# ----------------- Appel DataForSEO --------------------------
login, password = get_dfs_creds()
if not login or not password:
    st.warning("Ajoute DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD en variables d'env (ou secrets Streamlit).")
    st.stop()
headers = _auth_headers(login, password)

@st.cache_data(show_spinner=True, ttl=1800)
def fetch_ranked_keywords_and_metrics(target_root: str, location_code: int, language_code: str,
                                      limit_total: int = 5000,
                                      date_from: Optional[date] = None, date_to: Optional[date] = None,
                                      allow_fallback_live: bool = True):
    url = f"{API_BASE}/dataforseo_labs/google/ranked_keywords/live"
    def _run(with_dates: bool):
        rows_items, metrics, offset = [], None, 0
        while offset < limit_total:
            limit = min(1000, limit_total - offset)
            body = {
                "target": target_root.strip(),
                "location_code": location_code,
                "language_code": language_code,
                "ignore_synonyms": False,
                "include_clickstream_data": False,
                "load_rank_absolute": True,
                "limit": limit,
                "offset": offset
            }
            if with_dates and date_from and date_to:
                body["date_from"] = date_from.strftime("%Y-%m-%d")
                body["date_to"]   = date_to.strftime("%Y-%m-%d")
                body["historical_serp_mode"] = "as_is"

            data = safe_post(url, headers, [body], timeout=90)
            got = 0
            for task in (data.get("tasks") or []):
                for res in (task.get("result") or []):
                    if metrics is None:
                        metrics = ((res.get("metrics") or {}).get("organic") or {})
                    for it in (res.get("items") or []):
                        kw = ((it.get("keyword_data") or {}).get("keyword"))
                        rank = ((it.get("ranked_serp_element") or {}).get("rank_absolute"))
                        if kw is not None:
                            rows_items.append({"rk_keyword": kw, "rank_absolute": rank})
                            got += 1
            if got == 0: break
            offset += got
            time.sleep(0.12)

        items_df = pd.DataFrame(rows_items).drop_duplicates(subset=["rk_keyword"])
        if not items_df.empty:
            items_df["rk_norm2"] = items_df["rk_keyword"].apply(normalize_kw_strict)
        metrics_df = pd.DataFrame([metrics]) if isinstance(metrics, dict) else pd.DataFrame()
        return metrics_df, items_df

    if date_from and date_to:
        m, i = _run(with_dates=True)
        if (m.empty and i.empty) and allow_fallback_live:
            m2, i2 = _run(with_dates=False)
            return m2, i2, True
        return m, i, False
    m0, i0 = _run(with_dates=False)
    return m0, i0, False

# Période A
with st.spinner("Récupération (DataForSEO Labs)…"):
    metrics_df, rk_items, used_live_A = fetch_ranked_keywords_and_metrics(
        target_domain, LOCATION_CODE, LANGUAGE_CODE, LIMIT_RK,
        date_from=start_A, date_to=end_d
    )
if used_live_A:
    date_note.caption("Pas de données pour la période demandée — données 'live'.")

# Période B optionnelle
metrics_B, rk_items_B = pd.DataFrame(), pd.DataFrame()
if compare_prev:
    end_B   = start_A - timedelta(days=1)
    start_B = end_B - timedelta(days=window-1)
    with st.spinner("Récupération période précédente…"):
        metrics_B, rk_items_B, _ = fetch_ranked_keywords_and_metrics(
            target_domain, LOCATION_CODE, LANGUAGE_CODE, LIMIT_RK,
            date_from=start_B, date_to=end_B
        )

# ------------------ KPI (période A) --------------------------
pct_top10 = pct_11_20 = 0.0
total_count = 0
if not metrics_df.empty:
    m = metrics_df.iloc[0].to_dict()
    pos_1     = int(m.get("pos_1", 0) or 0)
    pos_2_3   = int(m.get("pos_2_3", 0) or 0)
    pos_4_10  = int(m.get("pos_4_10", 0) or 0)
    pos_11_20 = int(m.get("pos_11_20", 0) or 0)
    total_count = int(m.get("count", 0) or 0)
    if total_count > 0:
        pct_top10 = 100.0 * (pos_1 + pos_2_3 + pos_4_10) / total_count
        pct_11_20 = 100.0 * pos_11_20 / total_count

# -------- Matching items ↔ labels (pour radar & “best theme”) -
def match_items_to_labels(items: pd.DataFrame, labels: pd.DataFrame, thr: float = 0.6) -> pd.DataFrame:
    if items.empty or labels.empty: return pd.DataFrame()
    lab = labels.copy(); lab["tokens"] = lab["keyword"].apply(tok_norm)
    rk  = items.copy();  rk["tokens"]  = rk["rk_keyword"].apply(tok_norm)
    from collections import defaultdict
    bucket = defaultdict(list)
    for i, row in lab.iterrows():
        for t in list(row["tokens"])[:3]:
            bucket[t].append(i)
    rows = []
    for _, r in rk.iterrows():
        rtoks = r["tokens"]
        if not rtoks: continue
        cand_idx = set()
        for t in list(rtoks)[:3]:
            cand_idx.update(bucket.get(t, []))
        best_i, best_score = None, 0.0
        for i in (cand_idx or range(len(lab))):
            ltoks = lab.loc[i, "tokens"]

            if not ltoks: continue
            inter = len(rtoks & ltoks); union = len(rtoks | ltoks)
            jacc = inter / union if union else 0.0
            contains = min(inter / len(ltoks), inter / len(rtoks)) if inter >= 2 else 0.0
            score = max(jacc, contains)
            if score > best_score:
                best_score, best_i = score, i
        if best_i is not None and best_score >= thr:
            rows.append({"label_idx": best_i, "rank_absolute": r["rank_absolute"]})
    return pd.DataFrame(rows)

matches_A = match_items_to_labels(rk_items, labels_df, thr=0.6)
matches_B = match_items_to_labels(rk_items_B, labels_df, thr=0.6) if compare_prev else pd.DataFrame()

def compute_theme_top10(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty: return pd.DataFrame(columns=["theme","top10_ratio"])
    tmp = matches_df.copy()
    tmp["theme"] = tmp["label_idx"].apply(lambda i: primary_theme(labels_df.loc[i, "labels"]))
    return (tmp.assign(in_top10=tmp["rank_absolute"].le(10))
               .groupby("theme", as_index=False)["in_top10"].mean()
               .rename(columns={"in_top10":"top10_ratio"}))

theme_A = compute_theme_top10(matches_A)
theme_B = compute_theme_top10(matches_B) if compare_prev else pd.DataFrame()

# ----- KPI 3: best theme (période A)
best_theme = "—"
if not matches_A.empty:
    counts = matches_A.copy()
    counts["theme"] = counts["label_idx"].apply(lambda i: primary_theme(labels_df.loc[i, "labels"]))
    agg = (counts.assign(in_top10=counts["rank_absolute"].le(10))
                 .groupby("theme", as_index=False)
                 .agg(top10_ratio=("in_top10","mean"), n=("in_top10","size"))
                 .sort_values(["top10_ratio","n"], ascending=[False, False]))
    if not agg.empty:
        best_theme = str(agg.iloc[0]["theme"])

# -------------------- KPI CARDS -------------------------------
c1_css = """
<style>
div[data-testid="stMetric"], div[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    border: 0.5px solid #d1d5db;
}
div[data-testid="stMetric"] *,
div[data-testid="metric-container"] * {
    color: #000000 !important;
    font-size: 1.1rem;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem;
}
</style>
"""
st.markdown(c1_css, unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Couverture Top 10 (global)", f"{pct_top10:.1f} %", help=f"Période: {start_A} → {end_d}")
c2.metric("Potentiel 11–20 (global)", f"{pct_11_20:.1f} %", help=f"Période: {start_A} → {end_d}")
c3.metric("Thème le mieux couvert", best_theme)

# === VISUELS — Distribution ================================
import altair as alt
import plotly.graph_objects as go

st.divider()
st.subheader("Keyword distribution")
if not metrics_df.empty:
    m = metrics_df.iloc[0].to_dict()
    buckets = {
        "Top 1": int(m.get("pos_1", 0) or 0),
        "Top 2–3": int(m.get("pos_2_3", 0) or 0),
        "Top 4–10": int(m.get("pos_4_10", 0) or 0),
        "Top 11–20": int(m.get("pos_11_20", 0) or 0),
        "Top 21–50": int(m.get("pos_21_50", 0) or 0),
        "Top 51–100": int(m.get("pos_51_100", 0) or 0),
    }
    dist_df = pd.DataFrame({"bucket": list(buckets.keys()), "count": list(buckets.values())})
    total_count = int(m.get("count", 0) or 0)
    dist_df["share_%"] = (dist_df["count"] / max(total_count, 1) * 100).round(1)

    box = st.container()
    box.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] {
            background:#ffffff; border:0.5px solid #d1d5db;
            border-radius:8px; padding:1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with box:
        cc1, cc2 = st.columns([2,1])
        with cc1:
            chart = (
                alt.Chart(dist_df)
                   .mark_bar()
                   .encode(
                       x=alt.X("bucket:N", sort=["Top 1","Top 2–3","Top 4–10","Top 11–20","Top 21–50","Top 51–100"]),
                       y=alt.Y("count:Q"),
                       tooltip=["bucket","count","share_%"]
                   ).properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        with cc2:
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
else:
    st.caption("Pas de métriques globales disponibles pour dessiner la distribution.")

# ================== RADAR — Themes overview =================
st.subheader("Strengths and weaknesses per theme (label)")

themes_all = sorted(theme_A["theme"].unique().tolist() if not theme_A.empty else [])
if not themes_all:
    themes_all = sorted({primary_theme(v) for v in labels_df["labels"]})

theme_A_f = theme_A
theme_B_f = theme_B if compare_prev else pd.DataFrame()

if not theme_A_f.empty:
    radar_A = theme_A_f.assign(top10_pct=(theme_A_f["top10_ratio"]*100).round(1))
    axis = sorted(radar_A["theme"].unique().tolist())
    if compare_prev:
        radar_B = theme_B_f.assign(top10_pct=(theme_B_f["top10_ratio"]*100).round(1)) if not theme_B_f.empty else pd.DataFrame()
        axis = sorted(set(axis) | set(radar_B.get("theme", [])))
    r_A = radar_A.set_index("theme").reindex(axis)["top10_pct"].fillna(0).tolist()
    if compare_prev:
        r_B = radar_B.set_index("theme").reindex(axis)["top10_pct"].fillna(0).tolist() if not radar_B.empty else [0]*len(axis)
    else:
        r_B = [0]*len(axis)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_A,
            theta=axis,
            fill="toself",
            name=f"A ({start_A} → {end_d})",
            line_color="#1f77b4",
            fillcolor="rgba(31,119,180,0.3)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=r_B,
            theta=axis,
            fill="toself",
            name=f"B ({start_A - timedelta(days=window-1)} → {start_A - timedelta(days=1)})",
            line_color="#ff7f0e",
            fillcolor="rgba(255,127,14,0.3)",
        )
    )

    fig.update_layout(
        polar=dict(bgcolor="white", radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=20, r=20, t=10, b=20),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Pas assez de correspondances items ↔ labels pour construire le radar (overlap insuffisant).")

    # ========================= CONCURRENCE =========================
st.divider()
st.subheader("Competition analysis")

# ---------- 1) UI : labels + sélection de concurrents ----------
# options de labels/thèmes (comme pour le radar)
themes_all_comp = sorted(theme_A["theme"].unique().tolist() if not theme_A.empty else [])
if not themes_all_comp:
    themes_all_comp = sorted({primary_theme(v) for v in labels_df["labels"]})
default_themes_comp = [t for t in themes_all_comp if t != "Unlabeled"] or themes_all_comp

comp_main, comp_filters = st.columns([3,2])
with comp_main:
    st.caption("Concurrents (une URL ou un domaine par ligne)")
    comp_text = st.text_area(
        " ",
        value="",
        height=90
    )
with comp_filters:
    use_suggest = st.toggle("Suggérer", value=True, help="Ajoute les concurrents organiques DataForSEO (Top 10–15).")
    comp_sel_themes = st.multiselect(
        "Filtrer par labels (thèmes)",
        options=themes_all_comp,
        default=default_themes_comp,
        help="Les stats concurrence ne prennent en compte que ces thèmes.",
    )

# ---------- 2) DataForSEO : suggestion de concurrents ----------
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_competitors(target_root: str, location_code: int, language_code: str, limit:int=15) -> list[str]:
    url = f"{API_BASE}/dataforseo_labs/google/competitors_domain/live"
    payload = [{
        "target": target_root,
        "location_code": location_code,
        "language_code": language_code,
        "limit": limit
    }]
    data = safe_post(url, headers, payload, timeout=60)
    out = []
    for task in (data.get("tasks") or []):
        for res in (task.get("result") or []):
            for it in (res.get("items") or []):
                dom = (it.get("domain") or "").lower().strip()
                if dom and dom != target_root:
                    out.append(dom)
    # dédoublonne en gardant l'ordre
    dedup = []
    for d in out:
        if d not in dedup: dedup.append(d)
    return dedup

suggested = fetch_competitors(target_domain, LOCATION_CODE, LANGUAGE_CODE) if use_suggest else []
# parse saisie manuelle
manual = []
for line in (comp_text or "").splitlines():
    line = line.strip()
    if not line: continue
    # accepte url complète ou domaine
    try:
        host = urlparse(line if line.startswith("http") else "https://" + line).netloc.lower().strip()
        manual.append(host[4:] if host.startswith("www.") else host)
    except Exception:
        pass

# sélection finale (max 10, pour éviter trop d’appels API)
competitors = []
for d in manual + suggested:
    if d not in competitors and d != target_domain:
        competitors.append(d)
competitors = competitors[:10]

# feedback UI
if competitors:
    st.caption("Concurrents utilisés : " + ", ".join(competitors))
else:
    st.info("Ajoute des concurrents (ou active *Suggérer*).")
    st.stop()

# ---------- 3) Récup des keywords concurrents + attribution label ----------
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_comp_ranked_keywords(domain: str, location_code: int, language_code: str, limit: int = 1500):
    """Retourne DataFrame rk_keyword, rank_absolute pour un domaine donné."""
    url = f"{API_BASE}/dataforseo_labs/google/ranked_keywords/live"
    rows, offset = [], 0
    while offset < limit:
        step = min(1000, limit - offset)
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "ignore_synonyms": False,
            "include_clickstream_data": False,
            "load_rank_absolute": True,
            "limit": step,
            "offset": offset
        }]
        data = safe_post(url, headers, payload, timeout=90)
        got = 0
        for task in (data.get("tasks") or []):
            for res in (task.get("result") or []):
                for it in (res.get("items") or []):
                    kw = ((it.get("keyword_data") or {}).get("keyword"))
                    rank = ((it.get("ranked_serp_element") or {}).get("rank_absolute"))
                    if kw is not None:
                        rows.append({"rk_keyword": kw, "rank_absolute": rank})
                        got += 1
        if got == 0: break
        offset += got
        time.sleep(0.12)
    df = pd.DataFrame(rows).drop_duplicates(subset=["rk_keyword"])
    if not df.empty:
        df["rk_norm2"] = df["rk_keyword"].apply(normalize_kw_strict)
    return df

# pré-calculs labels (index pour matching)
lab_idx = labels_df.copy()
lab_idx["tokens"] = lab_idx["keyword"].apply(tok_norm)

from collections import defaultdict
bucket = defaultdict(list)
for i, row in lab_idx.iterrows():
    for t in list(row["tokens"])[:3]:
        bucket[t].append(i)

def best_label_idx_for_kw_tokens(rtoks:set, thr:float=0.6) -> Optional[int]:
    if not rtoks: return None
    cand_idx = set()
    for t in list(rtoks)[:3]:
        cand_idx.update(bucket.get(t, []))
    best_i, best_score = None, 0.0
    for i in (cand_idx or range(len(lab_idx))):
        ltoks = lab_idx.loc[i, "tokens"]

        if not ltoks:
            continue
        inter = len(rtoks & ltoks); union = len(rtoks | ltoks)
        jacc = inter/union if union else 0.0
        contains = min(inter/len(ltoks), inter/len(rtoks)) if inter >= 2 else 0.0
        score = max(jacc, contains)
        if score > best_score:
            best_score, best_i = score, i
    return best_i if (best_i is not None and best_score >= 0.6) else None

# calcule les stats par concurrent / thèmes sélectionnés
rows = []
for dom in competitors:
    dfc = fetch_comp_ranked_keywords(dom, LOCATION_CODE, LANGUAGE_CODE, limit=1500)
    if dfc.empty:
        rows.append({"competitor": dom, "keywords": 0, "top10_pct": 0.0})
        continue
    dfc["tokens"] = dfc["rk_keyword"].apply(tok_norm)
    # map vers theme
    lbl_idx = dfc["tokens"].apply(best_label_idx_for_kw_tokens)
    dfc["theme"] = lbl_idx.apply(lambda i: primary_theme(lab_idx.loc[i, "labels"]) if pd.notna(i) else "Unlabeled")
    # filtre thèmes choisis
    dfc = dfc[dfc["theme"].isin(comp_sel_themes or dfc["theme"].unique())]
    if dfc.empty:
        rows.append({"competitor": dom, "keywords": 0, "top10_pct": 0.0})
        continue
    # KPIs
    kcount = int(dfc.shape[0])
    top10 = float((dfc["rank_absolute"].le(10)).mean()*100.0)
    rows.append({"competitor": dom, "keywords": kcount, "top10_pct": round(top10,1)})

comp_df = pd.DataFrame(rows).sort_values("keywords", ascending=False)
if comp_df.empty:
    st.warning("Aucune donnée concurrent pour ces labels.")
    st.stop()

# part de visibilité (= part du volume de mots-clés filtrés)
comp_df["share_%"] = (comp_df["keywords"] / comp_df["keywords"].sum() * 100).round(1)
import plotly.express as px
colors = px.colors.qualitative.Plotly
box = st.container()
box.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] {
        background:#ffffff; border:0.5px solid #d1d5db;
        border-radius:8px; padding:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with box:
    g1, g2 = st.columns([2,1])
    with g1:
        bar = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("keywords:Q", title="Nombre de mots-clés (filtrés)"),
                y=alt.Y("competitor:N", sort='-x', title=None),
                color=alt.Color("competitor:N", scale=alt.Scale(range=colors), legend=None),
                tooltip=["competitor","keywords","top10_pct","share_%"]
            ).properties(height=420, title="Pages/keywords concurrents sur les mêmes thèmes")
        )
        st.altair_chart(bar, use_container_width=True)

    with g2:
        fig = go.Figure(
            data=[go.Pie(
                labels=comp_df["competitor"].tolist(),
                values=comp_df["keywords"].tolist(),
                hole=0.55,
                textinfo="percent+label",
                marker=dict(colors=colors[: len(comp_df)])
            )],
        )
        fig.update_layout(
            title="Share of visibility (mots-clés filtrés)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=420,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="v"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------- Footer / context ---------------------
rk_count = int(rk_items.shape[0]) if isinstance(rk_items, pd.DataFrame) else 0
overlap_A = int(matches_A.shape[0]) if isinstance(matches_A, pd.DataFrame) else 0
footer = (
    f"Domaine: **{target_domain}** • Location code: **{LOCATION_CODE}** • Lang code: **{LANGUAGE_CODE}** • "
    f"Période A: **{start_A} → {end_d}** (items: {rk_count}, matches: {overlap_A})"
)
if compare_prev:
    rk_count_B = int(rk_items_B.shape[0]) if isinstance(rk_items_B, pd.DataFrame) else 0
    overlap_B = int(matches_B.shape[0]) if isinstance(matches_B, pd.DataFrame) else 0
    footer += f" • Période B: **{start_A - timedelta(days=window-1)} → {start_A - timedelta(days=1)}** (items: {rk_count_B}, matches: {overlap_B})"
st.caption(footer)

if metrics_df.empty:
    st.warning("Aucune métrique globale renvoyée pour cette période (ou quota atteint).")
