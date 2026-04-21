# Semactic SEO Dashboard — Equans MVP

Interactive SEO dashboard built with Streamlit, reproducing the Semactic dashboard design for Equans Belgium NL.

## Features

- **Overview** — Native SEO KPIs (coverage, Top 10, position), competitive snapshot, label performance, priority labels
- **Share of Voice** — Panel-based SoV (est. clicks = vol × CTR by position), evolution, ranking, SoV by label
- **Connected Data** — Google Search Console & GA4 integration

## Data

Based on Equans Belgium NL keyword panel (320 keywords), with labels:
- Facility & Maintenance + Digital
- Equans Digital
- Carbon Shift
- Industry & Infrastructure
- ET-campagne

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology

- **Coverage metrics** (count-based): Ranked Coverage, Top 10 Coverage, Top 10 Share vs Competitors
- **Share of Voice** (traffic-based): SEO SoV = Σ est. clicks(domain, keyword) / Σ est. clicks(all domains, all keywords)
- Source: positions via SERP API, volumes via DataForSEO, CTR curve per position
