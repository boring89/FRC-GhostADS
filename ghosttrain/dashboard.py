"""Training dashboard (local web page).

    uv run python -m ghosttrain.dashboard runs/teacher3 [--port 8777] [--no-live] [--opponent baseline|self]

  http://localhost:8777/          training curves from <run>/log.jsonl
  http://localhost:8777/live      broadcast: field view + two robot POVs, red/blue scoreboard, 10 s result card
  http://localhost:8777/robot     one robot POV (choose 1-6) with detected AprilTags and status

The broadcast renders only while a page is open (no viewer = no rendering, no match clock). Training is not affected
except for the CPU/GPU share used while someone watches.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .broadcast import Broadcast

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>GhostADS training</title>
<style>
body{font:14px system-ui,sans-serif;background:#14161a;color:#e6e6e6;margin:0;padding:12px}
h1{font-size:18px;margin:0 0 8px} .row{display:flex;flex-wrap:wrap;gap:10px}
.card{background:#1d2026;border:1px solid #2c313a;border-radius:6px;padding:8px}
canvas{background:#14161a;display:block} .k{color:#8ab4f8} .small{color:#9aa0a6;font-size:12px}
table{border-collapse:collapse} td,th{padding:2px 8px;border-bottom:1px solid #2c313a;text-align:right;font-size:12px}
img{display:block;border-radius:4px;background:#000}
</style></head><body>
<h1>GhostADS teacher training <span class="small" id="stat"></span></h1>
<div class="row" id="charts"></div>
<div class="row" style="margin-top:10px">
  <div class="card"><div class="small">live match (newest checkpoint) &mdash; <span id="live"></span></div>
    <a href="/live" target="_blank"><img id="stream" src="/stream.mjpg" style="width:640px;height:360px" title="click: open fullscreen viewer"></a>
    <div class="small"><a href="/live" target="_blank" style="color:#8ab4f8">open fullscreen viewer</a> (F or double-click toggles fullscreen)</div></div>
  <div class="card" style="min-width:420px"><div class="small">latest anomalies (score &gt; 250 or fouls &gt; 40)</div>
    <table id="anom"></table></div>
</div>
<script>
const CH=[
 {t:"win rate vs baseline (EMA) - target 0.8",keys:[["winrate_baseline","#4caf50"],["win_self","#888"]],ref:[0.5,0.8],y:[0,1]},
 {t:"match score (blue=learner, red=opponent)",keys:[["score_blue","#4da3ff"],["score_red","#ff6b6b"],["score_median","#ddd"]]},
 {t:"fouls committed per alliance per match",keys:[["foul_mean","#ffb74d"]],log:true},
 {t:"goal entropy (ln13=2.56)",keys:[["ent_goal","#ba68c8"]]},
 {t:"approx KL / clip fraction",keys:[["kl","#4dd0e1"],["clipfrac","#f06292"]]},
 {t:"critic explained variance",keys:[["expl_var","#aed581"]],y:[-0.2,1]},
 {t:"agent-steps / s",keys:[["sps","#90a4ae"]]},
 {t:"mean step reward",keys:[["ret_mean","#fff176"]]},
];
const box=document.getElementById("charts");
CH.forEach((c,i)=>{const d=document.createElement("div");d.className="card";
 d.innerHTML='<div class="small">'+c.t+'</div><canvas id="c'+i+'" width="430" height="190"></canvas>';box.appendChild(d)});
function smooth(a,w){const o=[];let s=0,n=0;const q=[];for(const v of a){if(v==null||isNaN(v)){o.push(null);continue}
 q.push(v);s+=v;if(q.length>w)s-=q.shift();o.push(s/q.length)}return o}
function draw(i,rows){const c=CH[i],cv=document.getElementById("c"+i),g=cv.getContext("2d"),W=cv.width,H=cv.height;
 g.clearRect(0,0,W,H);const xs=rows.map(r=>r.it);const series=c.keys.map(([k,col])=>[smooth(rows.map(r=>r[k]),c.keys.length>1&&k.startsWith("score")?5:3),col,k]);
 let lo=1e18,hi=-1e18;for(const [s] of series)for(const v of s)if(v!=null){const u=c.log?Math.log10(1+Math.max(v,0)):v;lo=Math.min(lo,u);hi=Math.max(hi,u)}
 if(c.y){lo=c.y[0];hi=c.y[1]}if(!isFinite(lo)){return}if(hi-lo<1e-9){hi=lo+1}
 const x0=xs[0],x1=xs[xs.length-1]||1,px=x=>38+(x-x0)/Math.max(x1-x0,1)*(W-46),py=v=>H-18-((c.log?Math.log10(1+Math.max(v,0)):v)-lo)/(hi-lo)*(H-26);
 g.strokeStyle="#2c313a";g.fillStyle="#9aa0a6";g.font="10px sans-serif";
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=H-18-(H-26)*k/4;g.beginPath();g.moveTo(38,y);g.lineTo(W-8,y);g.stroke();
  g.fillText((c.log?Math.pow(10,v)-1:v).toPrecision(3),2,y+3)}
 g.fillText("it "+x0,38,H-4);g.fillText("it "+x1,W-60,H-4);
 for(const r of (c.ref||[])){g.strokeStyle="#ffffff44";g.setLineDash([4,4]);g.beginPath();g.moveTo(38,py(r));g.lineTo(W-8,py(r));g.stroke();g.setLineDash([])}
 for(const [s,col,k] of series){g.strokeStyle=col;g.lineWidth=1.5;g.beginPath();let on=false;
  s.forEach((v,j)=>{if(v==null){on=false;return}const X=px(xs[j]),Y=py(v);if(!on){g.moveTo(X,Y);on=true}else g.lineTo(X,Y)});g.stroke()}
 let lx=44;for(const [s,col,k] of series){g.fillStyle=col;const last=[...s].reverse().find(v=>v!=null);
  g.fillText(k+(last!=null?" "+last.toPrecision(3):""),lx,10);lx+=110}}
async function tick(){
 try{const rows=await (await fetch("/api/log?n=1500")).json();if(rows.length){CH.forEach((_,i)=>draw(i,rows));
  const r=rows[rows.length-1];document.getElementById("stat").textContent="iteration "+r.it+" | "+(r.steps/1e6).toFixed(1)+"M agent-steps | pool "+r.pool+" | shaping weight "+r.w_shape.toFixed(2)}}catch(e){}
 try{const a=await (await fetch("/api/anomalies")).json();document.getElementById("anom").innerHTML="<tr><th>it</th><th>opp</th><th>score b/r</th><th>fouls b/r</th></tr>"+
  a.map(x=>"<tr><td>"+x.it+"</td><td>"+x.opp+"</td><td>"+x.score.join(" / ")+"</td><td>"+x.fouls_committed.join(" / ")+"</td></tr>").join("")}catch(e){}
 try{const s=await (await fetch("/api/status")).json();document.getElementById("live").textContent=s.live}catch(e){}
}
tick();setInterval(tick,5000);
</script></body></html>"""


BROADCAST_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GhostADS live</title><style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
img{width:100vw;height:100vh;object-fit:contain;display:block}
</style></head><body>
<img id="s" src="/stream.mjpg">
<script>
const tg=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
addEventListener("keydown",e=>{if(e.key=="f"||e.key=="F")tg()});addEventListener("dblclick",tg);
</script></body></html>"""

ROBOT_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GhostADS robot view</title><style>
body{margin:0;background:#15171b;color:#ddd;font:14px sans-serif}
.bar{padding:8px 12px;display:flex;gap:8px;align-items:center}
.bar a{padding:6px 12px;border-radius:4px;background:#2b2f36;color:#ddd;text-decoration:none}
.bar a.on{background:#2a6fdb;color:#fff}
.wrap{display:flex;gap:16px;padding:8px 12px}
img{width:960px;height:720px;object-fit:contain;background:#000;border-radius:4px}
pre{background:#1d2026;padding:10px;border-radius:4px;min-width:340px;white-space:pre-wrap}
</style></head><body>
<div class="bar" id="bar"></div>
<div class="wrap"><img id="s"><pre id="st">loading...</pre></div>
<script>
const q=new URLSearchParams(location.search); const id=parseInt(q.get("id")||"1");
const bar=document.getElementById("bar");
for(let i=1;i<=6;i++){const a=document.createElement("a");a.href="/robot?id="+i;a.textContent="robot "+i+(i<=3?" (blue)":" (red)");if(i==id)a.className="on";bar.appendChild(a);}
document.getElementById("s").src="/robot.mjpg?id="+(id-1);
async function up(){try{const r=await (await fetch("/api/robot?id="+(id-1))).json();
document.getElementById("st").textContent=JSON.stringify(r,null,1);}catch(e){}}
setInterval(up,500);up();
</script></body></html>"""


def make_handler(run: Path, bc: Broadcast | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _rows(self, name, n):
            p = run / name
            if not p.exists():
                return []
            out = []
            for line in p.read_text().splitlines()[-n:]:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return out

        def _mjpeg(self, key, viewer_key):
            """Stream frames of `key` while this connection is open; counts as a viewer of `viewer_key`."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            bc.viewers[viewer_key] += 1
            seq = -1
            try:
                while True:
                    with bc.cond:
                        bc.cond.wait_for(lambda: bc.seq[key] != seq, timeout=5)
                        frame, seq = bc.frames[key], bc.seq[key]
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                    self.wfile.write(frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                bc.viewers[viewer_key] -= 1

        def do_GET(self):
            path, _, q = self.path.partition("?")
            params = dict(x.split("=") for x in q.split("&") if "=" in x)
            n = int(params.get("n", 1500))
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/live":
                self._send(BROADCAST_PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/robot":
                self._send(ROBOT_PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/log":
                self._send(json.dumps(self._rows("log.jsonl", n)).encode())
            elif path == "/api/anomalies":
                self._send(json.dumps(self._rows("anomalies.jsonl", 12)[::-1]).encode())
            elif path == "/api/status":
                self._send(json.dumps({"live": bc.info if bc else "live view disabled (--no-live)",
                                       "viewers": bc.viewers if bc else {}, "wins": bc.wins if bc else []}).encode())
            elif path == "/api/robot" and bc:
                r = int(params.get("id", 0))
                self._send(json.dumps(bc.status.get(r, {"robot": r + 1, "note": "no data yet"})).encode())
            elif path == "/stream.mjpg" and bc:
                self._mjpeg("bc", "bc")
            elif path == "/robot.mjpg" and bc:
                bc.robot_sel = int(params.get("id", 0))
                self._mjpeg("robot", "robot")
            else:
                self.send_error(404)

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="training output directory (contains log.jsonl / latest_net.pt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--opponent", default="baseline", choices=["baseline", "self"])
    a = ap.parse_args(argv)
    run = Path(a.run)
    bc = None if a.no_live else Broadcast(run, a.opponent)
    if bc:
        bc.start()
    try:
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(run, bc))
    except OSError as e:
        print(f"cannot listen on {a.host}:{a.port} ({e}); try --port <other>", file=sys.stderr)
        return 1
    print(f"dashboard on http://{a.host}:{a.port}  (curves /, broadcast /live, robot view /robot)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
