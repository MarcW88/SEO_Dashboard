import os
import json
from typing import List

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

@st.cache_data(show_spinner=False)
def _get_service():
    """Authenticate and return a Search Console service."""
    info = os.getenv("GSC_SERVICE_ACCOUNT_JSON", "")
    key_dict = None
    if info:
        try:
            key_dict = json.loads(info)
        except Exception:
            key_dict = None
    if key_dict is None:
        try:
            key_dict = dict(st.secrets.get("gsc_service_account", {}))
        except Exception:
            key_dict = None
    if not key_dict:
        return None
    creds = service_account.Credentials.from_service_account_info(key_dict, scopes=SCOPES)
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)

@st.cache_data(show_spinner=True)
def query_search_console(site_url: str, start_date: str, end_date: str,
                         dimensions: List[str] | None = None, row_limit: int = 25000) -> pd.DataFrame:
    """Run a searchanalytics.query and return a DataFrame."""
    service = _get_service()
    if service is None:
        return pd.DataFrame()
    dimensions = dimensions or ["date"]
    rows, start_row = [], 0
    while start_row < row_limit:
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "rowLimit": min(25000, row_limit - start_row),
            "startRow": start_row,
        }
        if body["rowLimit"] <= 0:
            break
        try:
            resp = service.searchanalytics().query(siteUrl=site_url, body=body).execute()
        except Exception:
            break
        rws = resp.get("rows", [])
        if not rws:
            break
        rows.extend(rws)
        start_row += len(rws)
        if len(rws) < body["rowLimit"]:
            break
    if not rows:
        cols = dimensions + ["clicks", "impressions", "ctr", "position"]
        return pd.DataFrame(columns=cols)
    records = []
    for r in rows:
        rec = {d: k for d, k in zip(dimensions, r.get("keys", []))}
        rec.update({
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
            "ctr": r.get("ctr", 0),
            "position": r.get("position", 0),
        })
        records.append(rec)
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
