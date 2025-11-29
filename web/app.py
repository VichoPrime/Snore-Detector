from flask import Flask, jsonify, request, Response
from collections import deque
import os, json

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
STATUS = os.path.join(DATA_DIR, "status.json")
EVENTS = os.path.join(DATA_DIR, "events.csv")
CMD    = os.path.join(DATA_DIR, "cmd.json")

app = Flask(__name__)

def tail_lines(path, n=50):
    try:
        with open(path, "r", encoding="utf-8") as f:
            dq = deque(f, maxlen=n)
        return [line.strip() for line in dq]
    except FileNotFoundError:
        return []

@app.get("/")
def index():
    html = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Snore Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:20px;background:#0b1220;color:#eef}
h1{margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px}
.card{background:#121b2e;border-radius:12px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.k{font-size:12px;opacity:.7}
.v{font-size:26px;font-weight:700}
.btn{padding:10px 14px;border-radius:10px;border:0;background:#2c74ff;color:#fff;font-weight:600;cursor:pointer}
.btn:disabled{opacity:.5;cursor:not-allowed}
table{width:100%;border-collapse:collapse}
th,td{padding:6px 8px;border-bottom:1px solid #1e2a44;font-size:13px}
th{opacity:.8;text-align:left}
.ok{color:#74ff9f}.bad{color:#ff7585}
footer{margin-top:18px;opacity:.6;font-size:12px}
</style></head><body>
<h1>Snore Detector — Dashboard</h1>

<div class="grid">
  <div class="card"><div class="k">score</div><div id="score" class="v">—</div></div>
  <div class="card"><div class="k">avg</div><div id="avg" class="v">—</div></div>
  <div class="card"><div class="k">rms</div><div id="rms" class="v">—</div></div>
  <div class="card"><div class="k">band</div><div id="band" class="v">—</div></div>
  <div class="card"><div class="k">estado</div><div id="state" class="v">—</div></div>
</div>

<div class="card" style="margin-bottom:16px">
  <button id="beep" class="btn">Beep de prueba</button>
  <span id="msg" style="margin-left:10px;font-size:12px;opacity:.8"></span>
</div>

<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <strong>Eventos recientes</strong><small id="evcount"></small>
  </div>
  <table><thead><tr><th>timestamp</th><th>score</th><th>avg</th><th>rms</th><th>band</th><th>thr</th></tr></thead>
  <tbody id="tbody"></tbody></table>
</div>

<footer>Dashboard para detector de ronquidos. Actualiza cada segundo.</footer>

<script>
async function fetchJSON(url){ const r=await fetch(url); if(!r.ok) throw new Error(r.status); return r.json(); }
async function fetchText(url){ const r=await fetch(url); if(!r.ok) throw new Error(r.status); return r.text(); }

async function refresh(){
  try{
    const st = await fetchJSON('/api/status');
    const el = id => document.getElementById(id);
    el('score').textContent = (st.score ?? '—');
    el('avg').textContent   = (st.avg ?? '—');
    el('rms').textContent   = (st.rms ?? '—');
    el('band').textContent  = (st.band ?? '—');
    el('state').textContent = st.state_on ? 'RONQUIDO' : 'normal';
    el('state').className   = st.state_on ? 'v bad' : 'v ok';
  }catch(e){}

  try{
    const t = await fetchText('/api/events?last=20');
    const lines = t.trim().split('\\n').filter(x=>x && !x.startsWith('timestamp'));
    document.getElementById('evcount').textContent = lines.length + ' mostrados';
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = lines.map(line=>{
      const [ts,score,avg,rms,band,lowmid,thr] = line.split(',');
      return `<tr><td>${ts}</td><td>${score}</td><td>${avg}</td><td>${rms}</td><td>${band}</td><td>${thr}</td></tr>`;
    }).join('');
  }catch(e){}
}
setInterval(refresh, 1000); refresh();

document.getElementById('beep').onclick = async ()=>{
  const msg = document.getElementById('msg');
  try{
    const r = await fetch('/api/beep', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({beep_ms:900})});
    msg.textContent = r.ok ? 'Beep enviado' : 'Error';
    setTimeout(()=> msg.textContent='', 2000);
  }catch(e){ msg.textContent='Error'; }
};
</script>
</body></html>
"""
    return Response(html, mimetype="text/html")

@app.get("/api/status")
def api_status():
    try:
        with open(STATUS, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({}), 200

@app.get("/api/events")
def api_events():
    n = int(request.args.get("last", 50))
    lines = tail_lines(EVENTS, n)
    return Response("\n".join(lines) + ("\n" if lines else ""), mimetype="text/plain")

@app.post("/api/beep")
def api_beep():
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = request.get_json(silent=True) or {}
    ms = int(payload.get("beep_ms", 900))
    ms = max(1, min(5000, ms))
    with open(CMD, "w", encoding="utf-8") as f:
        json.dump({"beep_ms": ms}, f)
    return jsonify({"ok": True, "beep_ms": ms})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
