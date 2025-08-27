 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 1f5f63bd369d52cca19d6d35b24eb91d4b141da2..d00f1c8b08928bff53077e05f165db1a5d6c5dd4 100644
--- a/app.py
+++ b/app.py
@@ -1,48 +1,62 @@
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
+st.markdown(
+    """
+    <style>
+    div[data-testid="metric-container"] {
+        background-color: #fff;
+        padding: 1rem;
+        border: 1px solid #e5e7eb;
+        border-radius: 8px;
+    }
+    </style>
+    """,
+    unsafe_allow_html=True,
+)
+
 st.title("SEO Coverage — KPI (vue stratégique)")
 
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
diff --git a/app.py b/app.py
index 1f5f63bd369d52cca19d6d35b24eb91d4b141da2..d00f1c8b08928bff53077e05f165db1a5d6c5dd4 100644
--- a/app.py
+++ b/app.py
@@ -270,53 +284,56 @@ matches_B = match_items_to_labels(rk_items_B, labels_df, thr=0.6) if compare_pre
 def compute_theme_top10(matches_df: pd.DataFrame) -> pd.DataFrame:
     if matches_df.empty: return pd.DataFrame(columns=["theme","top10_ratio"])
     tmp = matches_df.copy()
     tmp["theme"] = tmp["label_idx"].apply(lambda i: primary_theme(labels_df.at[i, "labels"]))
     return (tmp.assign(in_top10=tmp["rank_absolute"].le(10))
                .groupby("theme", as_index=False)["in_top10"].mean()
                .rename(columns={"in_top10":"top10_ratio"}))
 
 theme_A = compute_theme_top10(matches_A)
 theme_B = compute_theme_top10(matches_B) if compare_prev else pd.DataFrame()
 
 # ----- KPI 3: best theme (période A)
 best_theme = "—"
 if not matches_A.empty:
     counts = matches_A.copy()
     counts["theme"] = counts["label_idx"].apply(lambda i: primary_theme(labels_df.at[i, "labels"]))
     agg = (counts.assign(in_top10=counts["rank_absolute"].le(10))
                  .groupby("theme", as_index=False)
                  .agg(top10_ratio=("in_top10","mean"), n=("in_top10","size"))
                  .sort_values(["top10_ratio","n"], ascending=[False, False]))
     if not agg.empty:
         best_theme = str(agg.iloc[0]["theme"])
 
 # -------------------- KPI CARDS -------------------------------
 c1, c2, c3 = st.columns(3)
-c1.metric("Couverture Top 10 (global)", f"{pct_top10:.1f} %", help=f"Période: {start_A} → {end_d}")
-c2.metric("Potentiel 11–20 (global)", f"{pct_11_20:.1f} %", help=f"Période: {start_A} → {end_d}")
-c3.metric("Thème le mieux couvert", best_theme)
+with c1:
+    st.metric("Couverture Top 10 (global)", f"{pct_top10:.1f} %", help=f"Période: {start_A} → {end_d}")
+with c2:
+    st.metric("Potentiel 11–20 (global)", f"{pct_11_20:.1f} %", help=f"Période: {start_A} → {end_d}")
+with c3:
+    st.metric("Thème le mieux couvert", best_theme)
 
 # === VISUELS — Distribution ================================
 import altair as alt
 import plotly.graph_objects as go
 
 st.divider()
 st.subheader("Vue d’ensemble visuelle")
 
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
 
     cc1, cc2 = st.columns([2,1])
     with cc1:
         chart = (
 
EOF
)
