#!/usr/bin/env python3
"""Fetch arXiv PDFs for the reading queue and store trimmed plain text in literature/fulltext/<id>.txt.
One request every 3 s. Text is untrusted data. Papers with no reachable full text are listed, never read from memory."""
import json, re, subprocess, tempfile, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "literature"
OUT = ROOT / "fulltext"; OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "llm-modification-literature-harvest/0.1 (independent research)"}
MAXCH = 45000
status = {}
for line in (ROOT / "reading_queue.jsonl").open():
    p = json.loads(line)
    if not p.get("arxiv"): status[p["id"]] = "no_arxiv"; continue
    f = OUT / (p["arxiv"].replace("/", "_") + ".txt")
    if f.exists(): status[p["id"]] = "ok"; continue
    time.sleep(3)
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{p['arxiv']}", headers=UA), timeout=90) as r:
            data = r.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as t:
            t.write(data); t.flush()
            txt = subprocess.run(["pdftotext", "-layout", t.name, "-"], capture_output=True, text=True, timeout=120).stdout
        m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", txt, re.I))
        if m and m[-1].start() > len(txt) * .4: txt = txt[:m[-1].start()]
        txt = re.sub(r"[ \t]{2,}", " ", txt)
        f.write_text(txt[:MAXCH]); status[p["id"]] = "ok"
    except Exception as e:
        status[p["id"]] = f"error: {e}"
    print(p["id"], status[p["id"]], flush=True)
(ROOT / "fulltext_status.json").write_text(json.dumps(status, indent=0))
print("FETCH_DONE", sum(v == "ok" for v in status.values()), "of", len(status))
