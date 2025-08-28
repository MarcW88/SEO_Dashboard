---
title: SEO Dashboard
emoji: 🏃
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: 5.42.0
python_version: 3.11
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Screaming Frog export

The app accepts a Screaming Frog "Internal: All" export as CSV or Excel. The file must contain an `Address` column with the full URL of each crawled page. The first folder or parameter of the URL is used to guess an initial category, which can then be edited in the interface before generating default regex rules for Google Search Console.
