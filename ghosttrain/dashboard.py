"""Live training dashboard (local web page).

    uv run python -m ghosttrain.dashboard runs/t1 [--port 8777] [--no-live] [--opponent baseline|self]

Open http://localhost:8777 . Top: training curves from <run>/log.jsonl (refresh every 5 s). Bottom: a live match
played by the newest <run>/latest_net.pt (reloaded whenever the trainer saves), rendered with the Godot renderer
as an MJPEG stream. The live match uses one CPU core and a little GPU next to the training run.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

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


LIVE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>GhostADS live</title><style>
html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none}
img{width:100vw;height:100vh;object-fit:contain;display:block}
#hud{position:fixed;left:8px;bottom:6px;color:#bbb;font:12px sans-serif;text-shadow:0 0 3px #000;transition:opacity .6s}
</style></head><body>
<img id="s" src="/stream.mjpg"><div id="hud">F / double-click: fullscreen</div>
<script>
const tg=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
addEventListener("keydown",e=>{if(e.key=="f"||e.key=="F")tg()});addEventListener("dblclick",tg);
const hud=document.getElementById("hud");let tm;
addEventListener("mousemove",()=>{hud.style.opacity=1;document.body.style.cursor="default";clearTimeout(tm);
 tm=setTimeout(()=>{hud.style.opacity=0;document.body.style.cursor="none"},2500)});
setInterval(async()=>{try{hud.textContent=(await (await fetch("/api/status")).json()).live+"   |   F / double-click: fullscreen"}catch(e){}},2000);
</script></body></html>"""


class Live:
    """Background thread: plays matches with the newest checkpoint and publishes JPEG frames."""

    def __init__(self, run: Path, opponent="baseline", speed=1.0, seed=0, width=1280, height=720, fps=25.0, msaa=2,
                 jpeg_quality=88):
        self.run, self.opponent, self.speed, self.seed = run, opponent, speed, seed
        self.w, self.h, self.msaa, self.jq = width, height, msaa, jpeg_quality
        self.stride = int(np.clip(round(50.0 / fps), 1, 5))              # physics runs at 50 Hz (5 substeps / 0.1 s tick)
        self.fps = 50.0 / self.stride
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.info = "starting..."
        self.stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _publish(self, jpg):
        with self.cond:
            self.frame, self.seq = jpg, self.seq + 1
            self.cond.notify_all()

    def _loop(self):
        try:
            self._run()
        except Exception as e:                                         # keep the dashboard alive, show the reason
            import traceback
            self.info = f"live match stopped: {e!r}"
            traceback.print_exc()

    def _run(self):
        import cv2
        import torch

        from ghostrender import dr
        from ghostrender.client import RenderClient
        from ghostsim import SimCfg, load
        from ghostsim.robots import ALLIANCE, R

        from .net import TeacherNet
        from .obs import Encoder
        from .replay import spectator_rig
        from .vecenv import _Worker

        game = load("2025")
        enc = Encoder(game)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = TeacherNet(enc.n_goals, enc.G).to(dev)
        ckpt, mtime, rng = self.run / "latest_net.pt", 0.0, np.random.default_rng(self.seed)
        keys = ("self", "others", "pieces", "pmask", "faces", "lm", "gmask")
        cam = spectator_rig(game, self.w, self.h)
        rc = RenderClient()
        rc.configure(game, [cam], n_envs=1, width=self.w, height=self.h, max_cams=1, msaa=self.msaa)
        match_no = 0
        while not self.stop:
            while not ckpt.exists():
                self.info = f"waiting for {ckpt} (the trainer saves every few minutes)"
                time.sleep(3)
            mt = ckpt.stat().st_mtime
            sd = torch.load(ckpt, map_location=dev, weights_only=False)
            net.load_state_dict(sd["net"] if "net" in sd else sd); net.eval()
            mtime = mt
            match_no += 1
            w = _Worker(SimCfg(n_envs=1, sensors=False, seed=int(rng.integers(1 << 30))))
            w.trunc_at[:] = 10 ** 9
            w.set_ctrl(np.array([[False, self.opponent == "baseline"]]))
            o = w.reset()
            rc.set_appearance(0, dr.sample_appearance(rng, w.sim.prm.half[0], w.sim.prm.height[0], w.sim.ours["bumper_z_m"]),
                              {**dr.sample_dr(rng), "crowd_density": 0.6})
            ticks = int(round((w.sim.end_t - w.sim.t[0]) / 0.1))
            ck_t = time.strftime("%H:%M:%S", time.localtime(mt))
            snaps, count = [], [0]
            sim = w.sim

            def hook():                                           # keep every `stride`-th physics substep for rendering
                count[0] += 1
                if count[0] % self.stride == 0:
                    snaps.append((sim.render_state(), float(sim.score[0, 0]), float(sim.score[0, 1]), float(sim.t[0])))
            sim.substep_hook = hook
            frame_dt, deadline, shown, t_fps = self.stride * 0.02 / self.speed, time.time(), 0, time.time()
            scale = self.h / 360.0
            name2 = "BASELINE" if self.opponent == "baseline" else "RED(self)"
            for k in range(ticks):
                t = {key: torch.from_numpy(np.ascontiguousarray(o[key])).to(dev).reshape(R, *o[key].shape[2:]) for key in keys}
                t["glob"] = torch.from_numpy(o["glob"]).to(dev)[:, ALLIANCE].reshape(R, -1)
                with torch.no_grad():
                    logits, mu, _ = net(t)
                    goal = torch.distributions.Categorical(logits=logits).sample().cpu().numpy().reshape(1, R)
                    stick = np.clip((mu + net.log_std.exp() * torch.randn_like(mu)).cpu().numpy().reshape(1, R, 3), -1, 1)
                snaps.clear()
                res = w.step(stick, goal)
                o = res["obs"]
                for st, sb, sr, tt in snaps:
                    img, _ = rc.render(st, [(0, 0, 0)])
                    fr = np.ascontiguousarray(img[0][..., ::-1])
                    txt = f"BLUE(learner) {sb:.0f}  {name2} {sr:.0f}   t={tt:.0f}s"
                    cv2.putText(fr, txt, (int(10 * scale), int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (0, 0, 0), max(2, int(4 * scale)))
                    cv2.putText(fr, txt, (int(10 * scale), int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (255, 255, 255), max(1, int(2 * scale)))
                    self._publish(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, self.jq])[1].tobytes())
                    shown += 1
                    now = time.time()
                    if now - t_fps > 2:
                        self.info = (f"match {match_no}, checkpoint saved {ck_t}, t={tt:.0f}s, {self.w}x{self.h}, "
                                     f"{shown / (now - t_fps):.1f} fps (target {self.fps:.0f})")
                        shown, t_fps = 0, now
                    deadline += frame_dt
                    if deadline < now:                            # renderer cannot keep up: play slower than real time
                        deadline = now
                    else:
                        time.sleep(deadline - now)
                if res["done"].any() or self.stop:
                    break
            sim.substep_hook = None


def make_handler(run: Path, live: Live | None):
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

        def do_GET(self):
            path, _, q = self.path.partition("?")
            n = int(dict(x.split("=") for x in q.split("&") if "=" in x).get("n", 1500))
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/live":
                self._send(LIVE_PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/log":
                self._send(json.dumps(self._rows("log.jsonl", n)).encode())
            elif path == "/api/anomalies":
                self._send(json.dumps(self._rows("anomalies.jsonl", 12)[::-1]).encode())
            elif path == "/api/status":
                self._send(json.dumps({"live": live.info if live else "live view disabled (--no-live)"}).encode())
            elif path == "/stream.mjpg" and live:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        with live.cond:
                            live.cond.wait_for(lambda: live.seq != seq, timeout=5)
                            frame, seq = live.frame, live.seq
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                        self.wfile.write(frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
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
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the live match (1 = real time)")
    ap.add_argument("--width", type=int, default=1280, help="live view render width (default 1280x720)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=25.0, help="live view frame rate: 50/k for k=1..5 (50, 25, 16.7, 12.5, 10)")
    ap.add_argument("--msaa", type=int, default=2, choices=[0, 1, 2, 3], help="anti-aliasing: 0 off, 1 2x, 2 4x, 3 8x")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    a = ap.parse_args(argv)
    run = Path(a.run)
    live = None if a.no_live else Live(run, a.opponent, a.speed, width=a.width, height=a.height, fps=a.fps,
                                       msaa=a.msaa, jpeg_quality=a.jpeg_quality)
    if live:
        live.start()
    try:
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(run, live))
    except OSError as e:
        print(f"cannot listen on {a.host}:{a.port} ({e}); try --port <other>", file=sys.stderr)
        return 1
    print(f"dashboard on http://{a.host}:{a.port}  (run dir: {run})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
