#!/usr/bin/env python3
"""Curated reading batches. Fetches any missing arXiv text (3 s gap), writes a trimmed packet per paper: first 12k and last 3k characters."""
import json, re, subprocess, tempfile, time, urllib.request
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "literature"; FT = L / "fulltext"; B = L / "batches"; B.mkdir(exist_ok=True)
UA = {"User-Agent": "llm-modification-literature-harvest/0.1 (independent research)"}
BATCHES = {
 "B1_conditional": "2409.05907 2604.24693 2609.01878 2410.12299 2508.17621 2406.17563 2505.22572 2608.00218 2607.05790 2605.01167 2608.08383 2607.19806 2410.12877 2509.22739 2506.07022",
 "B2_prompt_compile": "2605.03907 2605.10664 2610.04967 2609.09054 2606.09563 2605.06225 2610.04283 2604.19018 2606.29522 2506.03292 2605.30076 2410.00487 2511.00617 2605.16591 2310.15916 2310.15213",
 "B3_reliability": "2407.12404 2505.22637 2602.17881 2602.06801 2606.07696 2602.06941 2610.16084 2602.01716 2606.06735 2610.04245 2507.11771 2501.17148 2406.11717 2411.09003 2312.06681 2604.08524",
 "B4_editing_closed_form": "2202.05262 2601.04600 2308.14761 2306.03819 2410.23054 2504.05343 2410.09338 2405.14768 2504.01196 2312.11795 2606.22357 2601.07309 2601.09398 2502.04043 2508.16082",
 "B5_composition": "2407.06483 2606.11262 2608.09490 2602.18628 2606.20638 2610.03199 2610.29110 2505.24262 2503.08099 2410.12937 2505.11611 2609.07037 2603.02237 2406.00045 1910.07104 2502.10966",
 "B6_adjacent": "2510.04309 2602.17560 2610.16325 2406.14801 2007.14823 2610.40757 2603.12378 2002.10585 2109.03554 1910.01526 2506.01303 2608.30231 2603.13347 2606.19379 2609.35207 2409.01308",
 "B7_lora_tools": "2106.09685 2311.12023 2507.05566 2502.00987 2405.13952 2605.16800 2607.19391 2507.10419 2505.10719 2303.03340 2606.07909 2503.21760 2609.28003 2305.11564 2502.05233",
 "B8_foundations_limits": "2411.04430 2611.01378 2504.00395 2603.25839 2310.01405 2306.03341 2308.10248 2608.10260 2601.01828 2609.31122 2605.12809 2605.30521 2506.13734 2511.04694 2502.15851 2305.08809",
}
meta = {}
for l in (L / "papers.jsonl").open():
    p = json.loads(l)
    if p.get("arxiv"): meta[p["arxiv"]] = p
missing = []
for name, ids in BATCHES.items():
    d = B / name; d.mkdir(exist_ok=True); idx = []
    for a in ids.split():
        f = FT / f"{a}.txt"
        if not f.exists():
            time.sleep(3.1)
            try:
                with urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{a}", headers=UA), timeout=90) as r: data = r.read()
                with tempfile.NamedTemporaryFile(suffix=".pdf") as t:
                    t.write(data); t.flush()
                    txt = subprocess.run(["pdftotext", "-layout", t.name, "-"], capture_output=True, text=True, timeout=120).stdout
                m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", txt, re.I))
                if m and m[-1].start() > len(txt) * .4: txt = txt[:m[-1].start()]
                f.write_text(re.sub(r"[ \t]{2,}", " ", txt)[:45000])
            except Exception as e:
                missing.append((a, str(e))); continue
        t = f.read_text(); pk = t if len(t) < 15000 else t[:12000] + "\n\n[... middle omitted from this packet ...]\n\n" + t[-3000:]
        (d / f"{a}.txt").write_text(pk); p = meta.get(a, {})
        idx.append({"arxiv": a, "title": p.get("title"), "year": p.get("year"), "authors": (p.get("authors") or [])[:6]})
    (d / "index.json").write_text(json.dumps(idx, indent=0)); print(name, len(idx))
print("MISSING", missing); print("BATCHES_DONE")
