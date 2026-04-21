"""
Fetch SERP positions for Equans + competitors via DataForSEO
Then calculate real SoV and export data for the dashboard.
"""
import pandas as pd
import requests
import base64
import time
import json

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
SITE = "equans.be"
COMPETITORS = ["spie.be", "honeywell.com", "vinci-energies.be", "siemens.be", "se.com"]
LANGUAGE = "be_nl"
LOCATION_CODE = 2056  # Belgium
LANGUAGE_CODE = "nl"

DATAFORSEO_LOGIN = "contact@marcwilliame.be"
DATAFORSEO_PASSWORD = "3aac73969a1d8db9"

EXCEL_PATH = "/Users/marc/Desktop/Dev/equans - be - nl - keywords_v2 - 21 apr (1).xlsx"
OUTPUT_PATH = "/Users/marc/Desktop/Dev/Dashboard_SEO/serp_data.json"

# CTR curve
CTR = {1: 0.30, 2: 0.15, 3: 0.10, 4: 0.07, 5: 0.05,
       6: 0.04, 7: 0.03, 8: 0.025, 9: 0.02, 10: 0.015}

SITE_DOMAIN = SITE.replace("www.", "")
ALL_DOMAINS = [SITE_DOMAIN] + COMPETITORS

def get_auth_header():
    creds = f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}"
    return {
        "Authorization": f"Basic {base64.b64encode(creds.encode()).decode()}",
        "Content-Type": "application/json",
    }

def analyze_serp(keyword):
    """Fetch SERP for one keyword, extract positions for all domains."""
    output = {"keyword": keyword}
    for d in ALL_DOMAINS:
        output[f"{d}_pos"] = None

    try:
        r = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
            headers=get_auth_header(),
            json=[{
                "keyword": keyword,
                "location_code": LOCATION_CODE,
                "language_code": LANGUAGE_CODE,
                "device": "desktop",
                "os": "windows",
                "depth": 100,
            }],
            timeout=90,
        )
        data = r.json()
        if data.get("status_code") != 20000:
            return output

        task = data.get("tasks", [{}])[0]
        if task.get("status_code") != 20000:
            return output

        result = task.get("result", [None])[0]
        if not result:
            return output

        for item in result.get("items", []) or []:
            if item.get("type") != "organic":
                continue
            domain = (item.get("domain") or "").lower().replace("www.", "")
            pos = item.get("rank_absolute", 0)

            for d in ALL_DOMAINS:
                col = f"{d}_pos"
                if output[col] is None and d.lower() in domain:
                    output[col] = pos

    except Exception as e:
        print(f"  ❌ Error [{keyword}]: {e}")

    return output

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    # 1. Load keywords
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip()

    kw_col = "Keyword" if "Keyword" in df.columns else "keyword"
    vol_col = "Volume" if "Volume" in df.columns else "volume"
    label_col = "Labels" if "Labels" in df.columns else "labels"

    keywords = df[kw_col].tolist()
    volumes = df[vol_col].tolist()
    labels_list = df[label_col].tolist()

    print(f"📊 {len(keywords)} keywords loaded")
    print(f"💰 Estimated cost: €{len(keywords) * 0.0075:.2f}")
    print(f"⏱️  Estimated time: ~{len(keywords) * 1.5 / 60:.0f} min")
    print(f"🎯 Domains: {SITE_DOMAIN} + {COMPETITORS}\n")

    # 2. Fetch SERP
    results = []
    for i, kw in enumerate(keywords):
        print(f"  [{i+1}/{len(keywords)}] {kw}", end="")
        res = analyze_serp(kw)
        results.append(res)
        # Show quick status
        positions = {d: res.get(f"{d}_pos") for d in ALL_DOMAINS}
        ranked = [f"{d}={p}" for d, p in positions.items() if p is not None]
        print(f"  → {', '.join(ranked) if ranked else 'no rankings'}")
        time.sleep(1)

    # 3. Calculate SoV
    print("\n══════════════════════════════════════")
    print("CALCULATING SOV...")

    domain_traffic = {d: 0 for d in ALL_DOMAINS}
    domain_top10 = {d: 0 for d in ALL_DOMAINS}
    label_data = {}

    for idx, res in enumerate(results):
        vol = volumes[idx] if idx < len(volumes) else 0
        vol = int(vol) if vol and vol == vol else 0
        lbl = str(labels_list[idx]) if idx < len(labels_list) and labels_list[idx] else "No label"

        for d in ALL_DOMAINS:
            pos = res.get(f"{d}_pos")
            if pos and pos <= 10:
                est_clicks = vol * CTR.get(int(pos), 0)
                domain_traffic[d] += est_clicks
                domain_top10[d] += 1
            # Per label
            for lb in lbl.split(", "):
                lb = lb.strip()
                key = (lb, d)
                if key not in label_data:
                    label_data[key] = {"traffic": 0, "top10": 0, "kw": 0}
                label_data[key]["kw"] += 1
                if pos and pos <= 10:
                    label_data[key]["traffic"] += vol * CTR.get(int(pos), 0)
                    label_data[key]["top10"] += 1

    total_traffic = sum(domain_traffic.values())

    # Print results
    print(f"\nTotal est. traffic across all domains: {total_traffic:,.0f}")
    print(f"\n{'Domain':<30} {'Est.Traffic':>12} {'SoV':>8} {'Top10 KW':>10}")
    print("-" * 65)
    for d in sorted(ALL_DOMAINS, key=lambda x: -domain_traffic[x]):
        sov = (domain_traffic[d] / total_traffic * 100) if total_traffic > 0 else 0
        print(f"{d:<30} {domain_traffic[d]:>12,.0f} {sov:>7.1f}% {domain_top10[d]:>10}")

    # 4. Build output JSON
    output = {
        "total_keywords": len(keywords),
        "total_traffic": total_traffic,
        "domains": {},
        "labels": {},
        "keywords": [],
    }

    for d in ALL_DOMAINS:
        sov = (domain_traffic[d] / total_traffic * 100) if total_traffic > 0 else 0
        output["domains"][d] = {
            "est_traffic": round(domain_traffic[d], 1),
            "sov_pct": round(sov, 2),
            "top10_count": domain_top10[d],
            "top10_coverage_pct": round(domain_top10[d] / len(keywords) * 100, 2),
        }

    # Label-level data
    all_labels = set(lb for lb, _ in label_data.keys())
    for lb in sorted(all_labels):
        output["labels"][lb] = {}
        for d in ALL_DOMAINS:
            key = (lb, d)
            if key in label_data:
                ld = label_data[key]
                lb_total = sum(label_data.get((lb, dd), {}).get("traffic", 0) for dd in ALL_DOMAINS)
                sov = (ld["traffic"] / lb_total * 100) if lb_total > 0 else 0
                output["labels"][lb][d] = {
                    "est_traffic": round(ld["traffic"], 1),
                    "sov_pct": round(sov, 2),
                    "top10_count": ld["top10"],
                    "kw_count": ld["kw"],
                }

    # Per keyword
    for idx, res in enumerate(results):
        vol = int(volumes[idx]) if idx < len(volumes) and volumes[idx] == volumes[idx] else 0
        kw_data = {
            "keyword": res["keyword"],
            "volume": vol,
            "label": str(labels_list[idx]) if idx < len(labels_list) and labels_list[idx] else "No label",
        }
        for d in ALL_DOMAINS:
            kw_data[f"{d}_pos"] = res.get(f"{d}_pos")
        output["keywords"].append(kw_data)

    # 5. Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Data saved to {OUTPUT_PATH}")
