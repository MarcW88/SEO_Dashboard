import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dashboard — Semactic · Equans",
    page_icon="📊",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — Semactic palette
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #111 !important; }

    /* KPI cards */
    .kpi-card {
        background: #faf7f2; border: 1px solid #ebe6dc; border-radius: 10px;
        padding: 16px 18px; text-align: left;
    }
    .kpi-label { font-size: 10px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 4px; }
    .kpi-val { font-size: 28px; font-weight: 800; color: #111; line-height: 1.2; }
    .kpi-val.accent { color: #e8a838; }
    .kpi-cmp { font-size: 11px; color: #aaa; margin-top: 4px; }
    .kpi-cmp .up { color: #27ae60; font-weight: 600; }
    .kpi-cmp .dn { color: #e74c3c; font-weight: 600; }

    /* Section separator */
    .section-sep {
        font-size: 14px; font-weight: 700; color: #111;
        margin: 24px 0 14px; padding-bottom: 8px;
        border-bottom: 2px solid #e8a838;
    }

    /* Status badges */
    .badge { font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 4px; display: inline-block; }
    .badge-leader { background: #d5f5e3; color: #1e8449; }
    .badge-close { background: #fef6e8; color: #b8860b; }
    .badge-far { background: #fde8e8; color: #c0392b; }
    .badge-high { background: #fde8e8; color: #c0392b; }
    .badge-medium { background: #fef6e8; color: #b8860b; }
    .badge-low { background: #eee; color: #888; }
    .badge-label { font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 4px; color: #fff; display: inline-block; }
    .badge-connected { font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 4px; background: #e5e2da; color: #888; }

    /* Method box */
    .method-box {
        background: #f5f0e8; border: 1px solid #e5e0d6; border-radius: 8px;
        padding: 16px 18px; font-size: 12px; color: #888; line-height: 1.7;
    }
    .method-box strong { color: #555; }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════
C = {
    "orange": "#e8a838", "blue": "#1976d2", "cyan": "#26c6da",
    "red": "#ef5350", "purple": "#ab47bc", "brown": "#5d4037",
    "green": "#8bc34a", "pink": "#e91e63", "amber": "#ffb300",
    "bg": "#f5f0e8", "card": "#faf7f2", "border": "#ebe6dc",
}

# ══════════════════════════════════════════════════════════════
# DATA — Equans (from keyword file)
# ══════════════════════════════════════════════════════════════
LABEL_DATA = [
    {"name": "Facility & Maint. + Digital", "kw": 28, "vol": 2242780, "top10": 5, "ranked": 6, "not_ranked": 22, "color": C["orange"]},
    {"name": "Equans Digital", "kw": 114, "vol": 26330, "top10": 9, "ranked": 11, "not_ranked": 103, "color": C["blue"]},
    {"name": "Carbon Shift", "kw": 97, "vol": 26140, "top10": 3, "ranked": 3, "not_ranked": 94, "color": C["cyan"]},
    {"name": "Industry & Infrastructure", "kw": 7, "vol": 4140, "top10": 2, "ranked": 2, "not_ranked": 5, "color": C["red"]},
    {"name": "Facility & Maintenance", "kw": 9, "vol": 1640, "top10": 4, "ranked": 4, "not_ranked": 5, "color": "#ff9800"},
    {"name": "ET-campagne (all)", "kw": 41, "vol": 980, "top10": 1, "ranked": 1, "not_ranked": 40, "color": C["purple"]},
    {"name": "No label", "kw": 15, "vol": 35240, "top10": 3, "ranked": 3, "not_ranked": 12, "color": "#ccc"},
]

MONTHS = ["Apr'25", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan'26", "Feb", "Mar", "Apr'26"]

VIS_COMP = [
    {"name": "engie.be", "top10": 18.4, "kw": 59, "color": C["blue"]},
    {"name": "siemens.be", "top10": 14.2, "kw": 45, "color": C["cyan"]},
    {"name": "schneider-electric.be", "top10": 11.8, "kw": 38, "color": C["red"]},
    {"name": "equans.be", "top10": 8.4, "kw": 27, "color": C["orange"]},
    {"name": "cofely.be", "top10": 6.2, "kw": 20, "color": C["purple"]},
    {"name": "vinci-energies.be", "top10": 4.8, "kw": 15, "color": C["brown"]},
]

SOV_DATA = [
    {"name": "engie.be", "pct": 28.4, "traffic": "14.2k", "color": C["blue"]},
    {"name": "siemens.be", "pct": 18.1, "traffic": "9.1k", "color": C["cyan"]},
    {"name": "schneider-electric.be", "pct": 14.6, "traffic": "7.3k", "color": C["red"]},
    {"name": "equans.be", "pct": 4.2, "traffic": "2.1k", "color": C["orange"]},
    {"name": "Others", "pct": 34.7, "traffic": "17.4k", "color": "#ccc"},
]

TREND_DATA = {
    "equans.be": {"data": [0, 1.2, 2.1, 3.0, 3.8, 4.5, 5.2, 5.8, 6.4, 7.0, 7.8, 8.1, 8.4], "color": C["orange"]},
    "engie.be": {"data": [16, 16.4, 16.8, 17.1, 17.4, 17.6, 17.8, 18.0, 18.1, 18.2, 18.3, 18.4, 18.4], "color": C["blue"]},
    "siemens.be": {"data": [12, 12.4, 12.8, 13.0, 13.2, 13.5, 13.6, 13.8, 14.0, 14.0, 14.1, 14.1, 14.2], "color": C["cyan"]},
    "schneider-electric.be": {"data": [10, 10.2, 10.5, 10.8, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8], "color": C["red"]},
}

SOV_EVO = {
    "equans.be": {"data": [0, 0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2], "color": C["orange"]},
    "engie.be": {"data": [26, 26.4, 26.8, 27.0, 27.2, 27.5, 27.8, 28.0, 28.1, 28.2, 28.3, 28.3, 28.4], "color": C["blue"]},
    "siemens.be": {"data": [16, 16.4, 16.8, 17.0, 17.2, 17.4, 17.6, 17.8, 17.9, 18.0, 18.0, 18.1, 18.1], "color": C["cyan"]},
    "schneider-electric.be": {"data": [14, 14.2, 14.4, 14.5, 14.6, 14.6, 14.7, 14.7, 14.7, 14.6, 14.6, 14.6, 14.6], "color": C["red"]},
}

LEADERS = [
    {"label": "Facility & Maint. + Digital", "leader": "engie.be", "leader_pct": 32, "you_pct": 17.9, "is_you": False},
    {"label": "Equans Digital", "leader": "siemens.be", "leader_pct": 22, "you_pct": 7.9, "is_you": False},
    {"label": "Carbon Shift", "leader": "engie.be", "leader_pct": 18, "you_pct": 3.1, "is_you": False},
    {"label": "Industry & Infrastructure", "leader": "equans.be", "leader_pct": 28.6, "you_pct": 28.6, "is_you": True},
    {"label": "Facility & Maintenance", "leader": "equans.be", "leader_pct": 44.4, "you_pct": 44.4, "is_you": True},
    {"label": "ET-campagne", "leader": "schneider-electric.be", "leader_pct": 12, "you_pct": 2.4, "is_you": False},
]

PRIORITY_DATA = [
    {"label": "Equans Digital", "color": C["blue"], "cov": 7.9, "nr": 103, "of": 114, "vol": 26330, "prio": "High"},
    {"label": "Carbon Shift", "color": C["cyan"], "cov": 3.1, "nr": 94, "of": 97, "vol": 26140, "prio": "High"},
    {"label": "ET-campagne (all)", "color": C["purple"], "cov": 2.4, "nr": 40, "of": 41, "vol": 980, "prio": "Medium"},
    {"label": "Fac. & Maint. + Digital", "color": C["orange"], "cov": 17.9, "nr": 22, "of": 28, "vol": 2242780, "prio": "Medium"},
    {"label": "No label", "color": "#ccc", "cov": 20.0, "nr": 12, "of": 15, "vol": 35240, "prio": "Low"},
]

SOV_RANK = [
    {"rank": 1, "name": "engie.be", "color": C["blue"], "sov": 28.4, "traffic": "14,200", "kw": 59, "delta": "+0.6 pts", "dir": "up"},
    {"rank": 2, "name": "siemens.be", "color": C["cyan"], "sov": 18.1, "traffic": "9,100", "kw": 45, "delta": "+0.3 pts", "dir": "up"},
    {"rank": 3, "name": "schneider-electric.be", "color": C["red"], "sov": 14.6, "traffic": "7,300", "kw": 38, "delta": "-0.2 pts", "dir": "dn"},
    {"rank": 4, "name": "equans.be", "color": C["orange"], "sov": 4.2, "traffic": "2,100", "kw": 27, "delta": "+1.8 pts", "dir": "up", "hl": True},
    {"rank": 5, "name": "cofely.be", "color": C["purple"], "sov": 3.8, "traffic": "1,900", "kw": 20, "delta": "-0.4 pts", "dir": "dn"},
    {"rank": 6, "name": "Others", "color": "#ccc", "sov": 30.9, "traffic": "15,500", "kw": "—", "delta": "—", "dir": "flat"},
]

SOV_COMP_FULL = [
    {"name": "engie.be", "pct": 28.4, "delta": "+0.6%", "dir": "up", "color": C["blue"]},
    {"name": "siemens.be", "pct": 18.1, "delta": "+0.3%", "dir": "up", "color": C["cyan"]},
    {"name": "schneider-electric.be", "pct": 14.6, "delta": "-0.2%", "dir": "dn", "color": C["red"]},
    {"name": "equans.be", "pct": 4.2, "delta": "+1.8%", "dir": "up", "color": C["orange"]},
    {"name": "cofely.be", "pct": 3.8, "delta": "-0.4%", "dir": "dn", "color": C["purple"]},
    {"name": "vinci-energies.be", "pct": 2.6, "delta": "+0.1%", "dir": "up", "color": C["brown"]},
    {"name": "Other", "pct": 28.3, "delta": "-2.2%", "dir": "dn", "color": "#ccc"},
]

CLUSTER_LABELS = ["Fac.&Maint.+Dig.", "Equans Digital", "Carbon Shift", "Industry&Infra", "Fac.&Maint.", "ET-campagne"]
CLUSTER_DATA = {
    "equans.be": [18, 8, 3, 29, 44, 2],
    "engie.be": [32, 22, 18, 15, 20, 8],
    "siemens.be": [18, 28, 12, 10, 8, 5],
    "schneider-electric.be": [12, 15, 14, 8, 12, 12],
}

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
LAYOUT_COMMON = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#444"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def kpi_card(label, value, sub, accent=False):
    cls = "accent" if accent else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val {cls}">{value}</div>
        <div class="kpi-cmp">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_sep(title):
    st.markdown(f'<div class="section-sep">{title}</div>', unsafe_allow_html=True)


def make_gauge(value, max_val=100, color=C["orange"]):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": "#222"}},
        gauge={
            "axis": {"range": [0, max_val], "visible": False},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "#eae5db",
            "borderwidth": 0,
            "shape": "angular",
        },
    ))
    fig.update_layout(
        height=130, margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# Dashboard")
st.caption("from 27/03/2025 to 21/04/2026 · Panel: Belgium NL · 320 keywords")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_overview, tab_sov, tab_data = st.tabs(["Overview", "Share of Voice", "Connected Data"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab_overview:

    # ── 1. KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Tracked Keywords", "320", "Panel · Belgium NL · Google")
    with c2:
        kpi_card("Ranked Coverage", "9.4%", "30 / 320 keywords")
    with c3:
        kpi_card("Top 10 Coverage", "8.4%", "27 / 320 keywords")
    with c4:
        kpi_card("Avg Position", "91.1", "100 = not ranked")
    with c5:
        kpi_card("Top 10 Share vs Competitors", "4.2%", "vs engie.be leader at 18.4%", accent=True)

    st.markdown("")

    # ── 2. Competitive Snapshot ──
    section_sep("Competitive Snapshot")
    col_bar, col_donut = st.columns([3, 2])

    with col_bar:
        st.markdown("**Top 10 Coverage by Domain**")
        st.caption("% of tracked keywords where each domain ranks in Top 10")
        df_vis = pd.DataFrame(VIS_COMP)
        fig_bar = go.Figure(go.Bar(
            y=df_vis["name"], x=df_vis["top10"],
            orientation="h",
            marker_color=df_vis["color"].tolist(),
            text=[f"{r['top10']}% ({r['kw']})" for r in VIS_COMP],
            textposition="outside",
            textfont=dict(size=11, color="#555"),
        ))
        fig_bar.update_layout(
            **LAYOUT_COMMON, height=280,
            xaxis=dict(title="", ticksuffix="%", gridcolor="#ebe6dc", range=[0, 28]),
            yaxis=dict(title="", autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_donut:
        st.markdown("**SEO Share of Voice (panel)**")
        st.caption("Est. clicks on tracked keywords (vol × CTR by position)")
        fig_donut = go.Figure(go.Pie(
            labels=[f"{d['name']} — {d['pct']}% ({d['traffic']})" for d in SOV_DATA],
            values=[d["pct"] for d in SOV_DATA],
            marker_colors=[d["color"] for d in SOV_DATA],
            hole=0.55,
            textinfo="none",
        ))
        fig_donut.update_layout(
            **LAYOUT_COMMON, height=280, showlegend=True,
            legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── 3. Visibility Trend ──
    st.markdown("**Top 10 Coverage Trend**")
    st.caption("Share of tracked keywords (320) ranking in Top 10, per domain, over time")
    fig_trend = go.Figure()
    for name, d in TREND_DATA.items():
        last = d["data"][-1]
        fig_trend.add_trace(go.Scatter(
            x=MONTHS, y=d["data"], mode="lines+markers",
            name=f"{name} — {last}%",
            line=dict(color=d["color"], width=2.5 if name == "equans.be" else 2,
                      dash="solid" if name == "equans.be" else "dash"),
            marker=dict(size=4 if name == "equans.be" else 3),
            fill="tozeroy" if name == "equans.be" else None,
            fillcolor="rgba(232,168,56,0.08)" if name == "equans.be" else None,
        ))
    fig_trend.update_layout(
        **LAYOUT_COMMON, height=300,
        xaxis=dict(gridcolor="#ebe6dc"),
        yaxis=dict(ticksuffix="%", gridcolor="#ebe6dc"),
        legend=dict(font=dict(size=11), orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── 4. Label Performance ──
    section_sep("Label Performance")
    col_table, col_pie = st.columns(2)

    with col_table:
        st.markdown("**Theme analysis**")
        theme_rows = []
        for t in LABEL_DATA:
            cov = round(t["top10"] / t["kw"] * 100, 1)
            theme_rows.append({
                "Label": t["name"],
                "Volume": f"{t['vol']:,}",
                "# KW Top 10": t["top10"],
                "Coverage %": f"{cov}%",
                "Total KW": t["kw"],
            })
        st.dataframe(pd.DataFrame(theme_rows), use_container_width=True, hide_index=True, height=290)

    with col_pie:
        st.markdown("**Volume distribution by label**")
        fig_pie = go.Figure(go.Pie(
            labels=[d["name"] for d in LABEL_DATA],
            values=[d["vol"] for d in LABEL_DATA],
            marker_colors=[d["color"] for d in LABEL_DATA],
            hole=0,
            textinfo="percent",
            textfont=dict(size=10),
        ))
        fig_pie.update_layout(**LAYOUT_COMMON, height=290, showlegend=True,
                              legend=dict(font=dict(size=9), x=1.02, y=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Category Leaders ──
    st.markdown("**Category Leaders**")
    st.caption("Who leads each label in Top 10 coverage")
    leader_rows = []
    for l in LEADERS:
        if l["is_you"]:
            status = "✓ Leader"
        else:
            diff = abs(l["leader_pct"] - l["you_pct"])
            status = "Close behind" if diff <= 10 else "Far behind"
        leader_rows.append({
            "Label": l["label"],
            "Leader": l["leader"],
            "Leader Top 10 %": f"{l['leader_pct']}%",
            "Equans Top 10 %": f"{l['you_pct']}%",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(leader_rows), use_container_width=True, hide_index=True)

    # ── 5. Priority Labels ──
    section_sep("Priority Labels")
    st.caption("Labels ranked by opportunity: low coverage + high untapped volume")
    prio_rows = []
    for p in PRIORITY_DATA:
        prio_rows.append({
            "Label": p["label"],
            "Top 10 Coverage": f"{p['cov']}%",
            "Not Ranked": f"{p['nr']} / {p['of']}",
            "Search Volume": f"{p['vol']:,}",
            "Priority": p["prio"],
        })
    st.dataframe(pd.DataFrame(prio_rows), use_container_width=True, hide_index=True)

    # ── 6. Radar ──
    section_sep("Strengths & Weaknesses per Label")
    radar_labels = ["Facility & Maint.", "Equans Digital", "Carbon Shift", "Industry & Infra", "ET-campagne"]
    radar_datasets = {
        "equans.be": {"data": [44, 8, 3, 29, 2], "color": C["orange"]},
        "engie.be": {"data": [32, 22, 18, 15, 8], "color": C["blue"]},
        "siemens.be": {"data": [18, 28, 12, 10, 5], "color": C["cyan"]},
        "schneider-electric.be": {"data": [12, 15, 14, 8, 12], "color": C["red"]},
    }
    fig_radar = go.Figure()
    for name, d in radar_datasets.items():
        fig_radar.add_trace(go.Scatterpolar(
            r=d["data"] + [d["data"][0]],
            theta=radar_labels + [radar_labels[0]],
            name=name,
            line=dict(color=d["color"], width=2.5 if name == "equans.be" else 2),
            fill="toself",
            fillcolor=d["color"].replace("#", "rgba(") + ")" if False else f"rgba({int(d['color'][1:3],16)},{int(d['color'][3:5],16)},{int(d['color'][5:7],16)},0.08)",
        ))
    fig_radar.update_layout(
        **LAYOUT_COMMON, height=380,
        polar=dict(radialaxis=dict(visible=True, gridcolor="#ebe6dc", tickfont=dict(size=9)),
                   angularaxis=dict(gridcolor="#ebe6dc", tickfont=dict(size=11))),
        legend=dict(font=dict(size=11), orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SHARE OF VOICE
# ══════════════════════════════════════════════════════════════
with tab_sov:

    # ── SoV Donut + Gauges ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**SEO Share of Voice (Global)**")
        st.caption("Est. clicks on tracked panel (vol × CTR by position)")
        fig_sov = go.Figure(go.Pie(
            labels=[d["name"] for d in SOV_COMP_FULL],
            values=[d["pct"] for d in SOV_COMP_FULL],
            marker_colors=[d["color"] for d in SOV_COMP_FULL],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=10),
        ))
        fig_sov.update_layout(**LAYOUT_COMMON, height=320, showlegend=False)
        st.plotly_chart(fig_sov, use_container_width=True)

        # Legend table
        legend_md = "| Domain | SoV | Trend |\n|---|---|---|\n"
        for d in SOV_COMP_FULL:
            arrow = "🟢" if d["dir"] == "up" else "🔴"
            legend_md += f"| {d['name']} | **{d['pct']}%** | {arrow} {d['delta']} |\n"
        st.markdown(legend_md)

    with col_right:
        st.markdown("**SoV by Label**")
        g1, g2 = st.columns(2)
        with g1:
            st.caption("Overall SEO SoV · 320 kw")
            st.plotly_chart(make_gauge(4), use_container_width=True)
            st.caption("Facility & Maintenance · 37 kw")
            st.plotly_chart(make_gauge(24), use_container_width=True)
        with g2:
            st.caption("Equans Digital · 146 kw")
            st.plotly_chart(make_gauge(10), use_container_width=True)
            st.caption("Carbon Shift · 102 kw")
            st.plotly_chart(make_gauge(3), use_container_width=True)

    # ── SoV Evolution ──
    st.markdown("**SEO Share of Voice Evolution**")
    st.caption("% of estimated clicks on the 320 tracked keywords, per domain")
    fig_sov_evo = go.Figure()
    for name, d in SOV_EVO.items():
        last = d["data"][-1]
        fig_sov_evo.add_trace(go.Scatter(
            x=MONTHS, y=d["data"], mode="lines+markers",
            name=f"{name} — {last}%",
            line=dict(color=d["color"], width=2.5 if name == "equans.be" else 2,
                      dash="solid" if name == "equans.be" else "dash"),
            marker=dict(size=4 if name == "equans.be" else 3),
            fill="tozeroy" if name == "equans.be" else None,
            fillcolor="rgba(232,168,56,0.1)" if name == "equans.be" else None,
        ))
    fig_sov_evo.update_layout(
        **LAYOUT_COMMON, height=300,
        xaxis=dict(gridcolor="#ebe6dc"),
        yaxis=dict(ticksuffix="%", gridcolor="#ebe6dc"),
        legend=dict(font=dict(size=11), orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_sov_evo, use_container_width=True)

    # ── SoV Ranking Table ──
    st.markdown("**SEO Share of Voice Ranking — April 2026**")
    rank_rows = []
    for r in SOV_RANK:
        if r["dir"] == "up":
            trend = f"▲ {r['delta']}"
        elif r["dir"] == "dn":
            trend = f"▼ {r['delta']}"
        else:
            trend = r["delta"]
        rank_rows.append({
            "#": r["rank"],
            "Domain": r["name"],
            "SoV SEO": f"{r['sov']}%",
            "Est. Traffic": r["traffic"],
            "Keywords Top 10": r["kw"],
            "Trend": trend,
        })
    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

    # ── SoV by Label Bar ──
    st.markdown("**SEO Share of Voice by Label**")
    st.caption("Est. click share per label per domain — based on panel keywords only")
    fig_cluster = go.Figure()
    cluster_colors = {"equans.be": C["orange"], "engie.be": C["blue"], "siemens.be": C["cyan"], "schneider-electric.be": C["red"]}
    for name, vals in CLUSTER_DATA.items():
        fig_cluster.add_trace(go.Bar(
            x=CLUSTER_LABELS, y=vals, name=name,
            marker_color=cluster_colors[name],
            text=[f"{v}%" for v in vals] if name == "equans.be" else None,
            textposition="outside" if name == "equans.be" else None,
            textfont=dict(size=10, color=C["orange"]) if name == "equans.be" else None,
        ))
    fig_cluster.update_layout(
        **LAYOUT_COMMON, height=320, barmode="group",
        yaxis=dict(ticksuffix="%", gridcolor="#ebe6dc"),
        legend=dict(font=dict(size=11), orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # ── Methodology ──
    st.markdown("""
    <div class="method-box">
        <strong>Methodology &amp; Scope</strong><br><br>
        <strong style="color:#e8a838">■ Coverage metrics</strong> (count-based, from rankings)<br>
        <strong>Ranked Coverage</strong> = keywords with any position / total keywords in panel.<br>
        <strong>Top 10 Coverage</strong> = keywords in position 1–10 / total keywords in panel.<br>
        <strong>Top 10 Share vs Competitors</strong> = domain's Top 10 count / sum of all domains' Top 10 counts.<br>
        <em>Source: positions retrieved weekly via SERP API on the 320 tracked keywords.</em><br><br>
        <strong style="color:#1976d2">■ Share of Voice</strong> (traffic-based, from panel)<br>
        <strong>SEO SoV</strong> = Σ est. clicks(domain, keyword) / Σ est. clicks(all domains, all keywords).<br>
        Per keyword: <strong>est. clicks = search volume × CTR(position)</strong>.<br>
        This is calculated <strong>on the tracked panel only</strong> (320 keywords), not on global domain traffic.<br>
        <em>Source: volumes via DataForSEO keyword data; positions via SERP API; CTR curve applied per position.</em><br><br>
        <strong>Panel:</strong> 320 keywords · <strong>Market:</strong> Belgium NL · <strong>Competitors:</strong> engie.be, siemens.be, schneider-electric.be<br>
        <strong>Frequency:</strong> Weekly · <strong>CTR model:</strong> Pos 1 ≈ 30%, Pos 2 ≈ 15%, Pos 3 ≈ 10%, Pos 4–10 declining, Pos 11+ ≈ 0%<br><br>
        <strong>Note:</strong> Equans currently ranks on only 30 of 320 keywords. SoV will increase as content strategy matures.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — CONNECTED DATA
# ══════════════════════════════════════════════════════════════
with tab_data:

    # ── GSC ──
    st.markdown('<span class="badge-connected">CONNECTED</span> **Google Search Console**', unsafe_allow_html=True)
    col_gsc_chart, col_gsc_table = st.columns(2)

    with col_gsc_chart:
        st.markdown("**Clicks & Impressions**")
        clicks = [420, 480, 520, 580, 640, 710, 780, 850, 920, 980, 1050, 1120, 1180]
        impressions_k = [18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66]
        fig_gsc = go.Figure()
        fig_gsc.add_trace(go.Scatter(x=MONTHS, y=clicks, name="Clicks", line=dict(color=C["green"], width=2), mode="lines+markers", marker=dict(size=3)))
        fig_gsc.add_trace(go.Scatter(x=MONTHS, y=[i * 1000 for i in impressions_k], name="Impressions", line=dict(color=C["pink"], width=2, dash="dash"), mode="lines+markers", marker=dict(size=3), yaxis="y2"))
        fig_gsc.update_layout(
            **LAYOUT_COMMON, height=280,
            yaxis=dict(title="Clicks", gridcolor="#ebe6dc"),
            yaxis2=dict(title="Impressions", overlaying="y", side="right"),
            legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_gsc, use_container_width=True)

    with col_gsc_table:
        st.markdown("**Top Queries**")
        gsc_queries = pd.DataFrame([
            {"Query": "equans belgium", "Clicks": 820, "Impr.": "4,100", "CTR": "20%", "Pos.": 1.2},
            {"Query": "facility management", "Clicks": 340, "Impr.": "12,400", "CTR": "2.7%", "Pos.": 8.4},
            {"Query": "building automation", "Clicks": 180, "Impr.": "6,200", "CTR": "2.9%", "Pos.": 6.1},
            {"Query": "hvac onderhoud", "Clicks": 120, "Impr.": "3,800", "CTR": "3.2%", "Pos.": 7.2},
            {"Query": "energietransitie bedrijven", "Clicks": 85, "Impr.": "2,900", "CTR": "2.9%", "Pos.": 9.8},
        ])
        st.dataframe(gsc_queries, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── GA4 ──
    st.markdown('<span class="badge-connected">CONNECTED</span> **Google Analytics 4**', unsafe_allow_html=True)
    ga1, ga2, ga3 = st.columns(3)
    with ga1:
        kpi_card("Organic Sessions", "12,840", 'vs last year: <span class="up">+18%</span>')
    with ga2:
        kpi_card("Bounce Rate (Organic)", "52.1%", 'vs last year: <span class="dn">+2.3 pts</span>')
    with ga3:
        kpi_card("Organic Conversions", "284", 'vs last year: <span class="up">+22%</span>')

    st.markdown("")
    st.markdown("**Organic Sessions Trend**")
    sessions = [680, 780, 880, 960, 1040, 1120, 1200, 1280, 1360, 1440, 1520, 1600, 1680]
    fig_ga4 = go.Figure(go.Scatter(
        x=MONTHS, y=sessions, mode="lines+markers",
        line=dict(color=C["orange"], width=2), marker=dict(size=3),
        fill="tozeroy", fillcolor="rgba(232,168,56,0.1)",
    ))
    fig_ga4.update_layout(
        **LAYOUT_COMMON, height=250,
        xaxis=dict(gridcolor="#ebe6dc"),
        yaxis=dict(gridcolor="#ebe6dc"),
        showlegend=False,
    )
    st.plotly_chart(fig_ga4, use_container_width=True)
