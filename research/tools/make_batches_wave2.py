#!/usr/bin/env python3
"""Wave-2 reading batches: hand-picked from the unread, full-text part of reading_queue.jsonl (embedding triage pulled in much off-topic
theory; C1-C3 (48) are the ones about LLM steering, tool/memory behaviour, editing and composition; C4 (16) came from a title search for tool-decision internals that triage missed). Packet = first 12k + last 3k chars."""
import json, re, subprocess, tempfile, time, urllib.request
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "literature"; FT = L / "fulltext"; B = L / "batches"
BATCHES = {
 "C1_steering_methods": "2609.35591 2501.11036 2606.08454 2501.09929 2609.32530 2607.25270 2506.06686 2602.04428 2510.13285 2602.16080 2606.15092 2510.26243 2312.03813 2411.11296 2503.00177 2609.30405",
 "C2_tools_memory_repair": "2509.16268 2601.05366 2602.13258 2602.17022 2607.22556 2112.09737 2605.05732 2609.36138 2604.12820 2207.11240 2605.30621 2609.22712 2505.16526 2405.00301 2501.16378 2601.06109",
 "C4_tool_decision_internals": "2605.07990 2608.25198 2610.09624 2605.09252 2605.14038 2602.04935 2607.11183 2606.30449 2605.00737 2606.25605 2603.09297 2602.07213 2404.19705 2504.20168 2601.16280 2609.25678",
 "C3_behaviour_composition": "2610.20887 2505.11615 2608.30986 2610.29555 2604.14463 2609.06289 2608.15687 2311.13534 2509.19476 1705.04228 2610.32581 2505.18356 1701.06538 2501.16476 2512.02689 2610.40511",
}
meta = {p["arxiv"]: p for p in map(json.loads, (L / "papers.jsonl").open()) if p.get("arxiv")}
missing = []
for name, ids in BATCHES.items():
    d = B / name; d.mkdir(exist_ok=True); idx = []
    for a in ids.split():
        f = FT / f"{a}.txt"
        if not f.exists():
            time.sleep(3.1)
            UA = {"User-Agent": "llm-modification-literature-harvest/0.1 (independent research)"}
            try:
                with urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{a}", headers=UA), timeout=90) as r: data = r.read()
            except Exception as e:
                missing.append((a, str(e))); continue
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
                tf.write(data); tf.flush(); txt = subprocess.run(["pdftotext", "-layout", tf.name, "-"], capture_output=True, text=True, timeout=120).stdout
            m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", txt, re.I))
            if m and m[-1].start() > len(txt) * .4: txt = txt[:m[-1].start()]
            f.write_text(re.sub(r"[ \t]{2,}", " ", txt)[:45000])
        t = f.read_text(); pk = t if len(t) < 15000 else t[:12000] + "\n\n[... middle omitted from this packet ...]\n\n" + t[-3000:]
        (d / f"{a}.txt").write_text(pk); p = meta.get(a, {})
        idx.append({"arxiv": a, "title": p.get("title"), "year": p.get("year"), "authors": (p.get("authors") or [])[:6]})
    (d / "index.json").write_text(json.dumps(idx, indent=0)); print(name, len(idx))
print("MISSING", missing)
