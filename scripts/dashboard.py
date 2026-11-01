"""Read-only training dashboard served on the local network. Uses only the standard library."""

import json
import re
import secrets
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TOKEN_FILE = ARTIFACTS / "dashboard_token"
PORT = 8765
ROUNDS = 4
BASELINE = 0.395
SEEDS = [42, 7, 123]
# Typical CPU round times, so slower MPS rounds do not inflate the estimate.
TYPICAL_ROUND_SECONDS = {"distilroberta_utterance": 360, "distilroberta_context3": 600,
                         "distilroberta_context3_speakers": 660, "roberta_context3": 1090}
DESIGNS = [
    ("distilroberta_utterance", "Single line"),
    ("distilroberta_context3", "With history"),
    ("distilroberta_context3_speakers", "With speakers"),
    ("roberta_context3", "roberta-base"),
]


def run(*command):
    return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout


def machine():
    power = run("pmset", "-g", "batt")
    percent = re.search(r"(\d+)%", power)
    swap = re.search(r"used = ([\d.]+)M", run("sysctl", "vm.swapusage"))
    return {
        "training": "scripts/guide.sh train" in run("ps", "-axo", "command"),
        "battery": int(percent.group(1)) if percent else None,
        "charging": "AC Power" in power,
        "swap_gb": round(float(swap.group(1)) / 1024, 1) if swap else None,
    }


def read_run(name, seed):
    folder = ARTIFACTS / name / f"seed_{seed}"
    config, history_file, summary = folder / "config.json", folder / "history.json", folder / "summary.json"
    history = []
    # A restarted run keeps its old history file until the first round ends.
    current = config.exists() and history_file.exists() and (
        history_file.stat().st_mtime >= config.stat().st_mtime or (folder / "resume.pt").exists())
    if current or summary.exists():
        history = json.loads(history_file.read_text())
    started = max((path.stat().st_mtime for path in (config, history_file) if path.exists()), default=None)
    return {"history": history, "finished": summary.exists(), "started": started if current or history == [] else None}


def round_seconds(runs, name):
    same = [row["seconds"] for item in runs if item["name"] == name for row in item["history"]]
    typical = TYPICAL_ROUND_SECONDS[name]
    return min(same[-1], typical) if same else typical


def status():
    runs = [{"name": name, "label": label, "seed": seed, **read_run(name, seed)}
            for seed in SEEDS for name, label in DESIGNS]
    state = machine()
    active = next((item for item in runs if not item["finished"]), None)
    progress, remaining = None, 0.0
    for item in runs:
        seconds = round_seconds(runs, item["name"])
        if not item["finished"] and seconds:
            remaining += (ROUNDS - len(item["history"])) * seconds
    if active and state["training"] and active["started"]:
        seconds = round_seconds(runs, active["name"])
        elapsed = time.time() - active["started"]
        if seconds:
            progress = min(elapsed / seconds, 0.99)
            remaining = max(remaining - elapsed, 0)
    rounds = {}
    for name, label in DESIGNS:
        values = [[] for _ in range(ROUNDS)]
        for item in runs:
            if item["name"] == name:
                for row in item["history"][:ROUNDS]:
                    values[row["epoch"] - 1].append(row["macro_f1"])
        rounds[label] = [sum(group) / len(group) if group else None for group in values]
    return {
        **state, "now": time.time(), "baseline": BASELINE, "rounds_per_run": ROUNDS, "round_progress": progress,
        "remaining_seconds": remaining if state["training"] else None, "rounds": rounds,
        "finished_runs": sum(item["finished"] for item in runs), "total_runs": len(runs),
        "active": runs.index(active) if active else None,
        "runs": [{"label": item["label"], "seed": item["seed"], "finished": item["finished"],
                  "rounds_done": len(item["history"]),
                  "best": max((row["macro_f1"] for row in item["history"]), default=None)} for item in runs],
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#1a1a19"><title>MELD Training</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--line:#e4e3de;--text:#0b0b0b;--muted:#52514e;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#7a5af8}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--card:#232322;--line:#3a3a37;
--text:#f0efec;--muted:#c3c2b7;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9580ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:16px/1.4 -apple-system,system-ui,sans-serif;
padding:max(16px,env(safe-area-inset-top)) 16px 32px}
main{max-width:560px;margin:0 auto}
h1{font-size:22px;margin:0}
h2{font-size:15px;margin:0 0 8px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:4px}
.pill{font-size:14px;font-weight:600;border:1px solid var(--line);border-radius:999px;padding:4px 10px;
white-space:nowrap}
.muted{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-top:12px}
.hero{font-size:30px;font-weight:700;line-height:1.15}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin:10px 0 6px}
.bar i{display:block;height:100%;background:var(--s1);border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px}
.tile b{display:block;font-size:18px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:0 0 6px}
.legend span{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:50%}
svg{display:block;width:100%;height:auto;touch-action:manipulation}
svg text{font:11px -apple-system,system-ui,sans-serif;fill:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{font-size:12px;color:var(--muted);text-align:left;font-weight:500}
th,td{padding:7px 4px;border-top:1px solid var(--line)}
th{border-top:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.now td{font-weight:600}
</style></head><body><main>
<div class="top"><h1>MELD Training</h1><div class="pill" id="pill">Connecting…</div></div>
<div class="muted" id="updated">Waiting for the Mac</div>
<section class="card"><div class="muted" id="runlabel"></div><div class="hero" id="hero">–</div>
<div class="bar"><i id="bar" style="width:0"></i></div><div class="muted" id="sub"></div></section>
<section class="tiles">
<div class="tile"><span class="muted">Time left</span><b id="left">–</b></div>
<div class="tile"><span class="muted">Battery</span><b id="battery">–</b></div>
<div class="tile"><span class="muted">Swap used</span><b id="swap">–</b></div></section>
<section class="card"><h2>Score by round (macro-F1, higher is better)</h2>
<div class="legend" id="legend"></div><div id="chart"></div>
<div class="muted" id="read">Tap the chart to see a round's scores.</div></section>
<section class="card"><h2>All runs</h2><table><thead><tr><th>#</th><th>Design</th><th>Seed</th>
<th class="n">Best</th><th>Status</th></tr></thead><tbody id="runs"></tbody></table></section>
</main><script>
const colors={"Single line":"var(--s1)","With history":"var(--s2)","With speakers":"var(--s3)","roberta-base":"var(--s4)"};
const $=id=>document.getElementById(id);
const hours=s=>{const m=Math.round(s/60);return m<60?m+" min":Math.floor(m/60)+"h "+String(m%60).padStart(2,"0")+"m"};
let latest=null;
function readout(i){if(!latest)return;
 const parts=Object.entries(latest.rounds).filter(([,v])=>v[i]!=null).map(([k,v])=>k+" "+v[i].toFixed(3));
 $("read").textContent="Round "+(i+1)+": "+(parts.join(" · ")||"no scores yet")}
function chart(d){
 const W=340,H=200,L=36,R=84,T=12,B=26,rows=Object.entries(d.rounds),all=rows.flatMap(([,v])=>v).filter(v=>v!=null);
 const lo=Math.floor((Math.min(d.baseline,...all)-.01)*50)/50,hi=Math.ceil((Math.max(d.baseline,...all)+.01)*50)/50;
 const x=i=>L+i*(W-L-R)/(d.rounds_per_run-1),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Macro-F1 by training round for each design">`;
 for(let k=0;k<3;k++){const v=lo+k*(hi-lo)/2;
  s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`}
 for(let i=0;i<d.rounds_per_run;i++)s+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">Round ${i+1}</text>`;
 s+=`<line x1="${L}" x2="${W-R}" y1="${y(d.baseline)}" y2="${y(d.baseline)}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 4"/>`;
 s+=`<text x="${W-R+6}" y="${y(d.baseline)+4}">Old best</text>`;
 const ends=[];
 for(const [name,v] of rows){const pts=v.map((p,i)=>p==null?null:[x(i),y(p)]).filter(Boolean);if(!pts.length)continue;
  s+=`<polyline fill="none" stroke="${colors[name]}" stroke-width="2" stroke-linejoin="round" points="${pts.map(p=>p.join(",")).join(" ")}"/>`;
  for(const p of pts)s+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${colors[name]}" stroke="var(--card)" stroke-width="2"/>`;
  const e=pts[pts.length-1];ends.push({name,x:e[0],y:e[1]})}
 ends.sort((a,b)=>a.y-b.y);ends.forEach((e,i)=>{if(i&&e.y-ends[i-1].y<13)e.y=ends[i-1].y+13;
  s+=`<text x="${e.x+8}" y="${e.y+4}">${e.name}</text>`});
 const step=(W-L-R)/(d.rounds_per_run-1);
 for(let i=0;i<d.rounds_per_run;i++)s+=`<rect x="${x(i)-step/2}" y="0" width="${step}" height="${H}" fill="transparent" data-i="${i}"/>`;
 $("chart").innerHTML=s+"</svg>";
 $("chart").querySelectorAll("rect").forEach(r=>r.onclick=()=>readout(+r.dataset.i));
 $("legend").innerHTML=rows.map(([k])=>`<span><i style="background:${colors[k]}"></i>${k}</span>`).join("")}
function render(d){latest=d;
 const a=d.active==null?null:d.runs[d.active],done=d.finished_runs==d.total_runs;
 $("pill").textContent=done?"✓ Finished":d.training?"▶ Running":"■ Stopped";
 $("updated").textContent="Updated "+new Date(d.now*1000).toLocaleTimeString([], {hour:"numeric",minute:"2-digit",second:"2-digit"});
 if(a){const round=Math.min(a.rounds_done+1,d.rounds_per_run),share=d.round_progress??0;
  $("runlabel").textContent="Run "+(d.active+1)+" of "+d.total_runs+" · "+a.label+" · seed "+a.seed;
  $("hero").textContent="Round "+round+" of "+d.rounds_per_run;
  $("bar").style.width=Math.round(share*100)+"%";
  $("sub").textContent=d.training?(d.round_progress==null?"Measuring speed":"About "+Math.round(share*100)+"% through this round (estimate)"):"Training is not running";
 }else{$("runlabel").textContent="All runs";$("hero").textContent="Complete";$("bar").style.width="100%";$("sub").textContent=""}
 $("left").textContent=d.remaining_seconds==null?"–":hours(d.remaining_seconds);
 $("battery").textContent=d.battery==null?"–":d.battery+"%"+(d.charging?" ⚡":"");
 $("swap").textContent=d.swap_gb==null?"–":d.swap_gb+" GB";
 $("runs").innerHTML=d.runs.map((r,i)=>`<tr class="${i===d.active?"now":""}"><td>${i+1}</td><td>${r.label}</td><td>${r.seed}</td>
<td class="n">${r.best==null?"–":r.best.toFixed(3)}</td><td>${r.finished?"Done":i===d.active?(d.training?"Running":"Stopped"):"Waiting"}</td></tr>`).join("");
 chart(d)}
async function tick(){try{const r=await fetch("status.json",{cache:"no-store"});render(await r.json())}
 catch{$("pill").textContent="⚠ No connection";$("updated").textContent="Can't reach the Mac. Check it's on the same Wi-Fi."}}
tick();setInterval(tick,20000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    token = ""

    def do_GET(self):
        if self.path == f"/{self.token}/":
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        elif self.path == f"/{self.token}/status.json":
            body, kind = json.dumps(status()).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(9))
    Handler.token = TOKEN_FILE.read_text().strip()
    address = run("ipconfig", "getifaddr", "en0").strip() or "localhost"
    print(f"Open on your iPhone: http://{address}:{PORT}/{Handler.token}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
