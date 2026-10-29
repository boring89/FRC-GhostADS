"""The /live page: the 1920x1080 MJPEG video with an official-style HTML overlay (scoreboard, tile labels, result
card), fed by /api/live (Broadcast.live). No sponsor or FIRST artwork. Sizes are in 1920x1080 units scaled with the
video (--u), so the overlay stays aligned and sharp at any window size."""

LIVE_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>GhostADS live</title><style>
:root{--u:1px;--blue:#1f4fd1;--blue2:#173ea8;--blueL:#2f63ea;--red:#d42a2a;--red2:#a91d1d;--redL:#e83b3b;--ink:#11141a}
html,body{margin:0;height:100%;background:#000;overflow:hidden;font-family:"Inter","Segoe UI","Roboto","Noto Sans","DejaVu Sans","Helvetica Neue",Arial,sans-serif}
.stage{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);aspect-ratio:16/9;width:min(100vw,177.78vh);background:#000;overflow:hidden}
.video{position:absolute;inset:0;width:100%;height:100%;display:block}
.num{font-variant-numeric:tabular-nums;font-weight:800}
/* scoreboard */
.sb{position:absolute;top:calc(22*var(--u));left:50%;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;filter:drop-shadow(0 calc(6*var(--u)) calc(14*var(--u)) rgba(0,0,0,.55))}
.title{background:rgba(10,12,18,.88);color:#e8ecf4;font-size:calc(20*var(--u));letter-spacing:.08em;padding:calc(6*var(--u)) calc(26*var(--u));border-radius:calc(6*var(--u)) calc(6*var(--u)) 0 0;font-weight:600;white-space:nowrap}
.row{display:flex;align-items:stretch;height:calc(104*var(--u))}
.name{width:calc(470*var(--u));display:flex;flex-direction:column;justify-content:center;color:#fff;padding:0 calc(26*var(--u))}
.name .al{font-size:calc(17*var(--u));letter-spacing:.14em;opacity:.85;font-weight:600}
.name .nm{font-size:calc(40*var(--u));font-weight:800;letter-spacing:.03em;line-height:1.1}
.name .rb{font-size:calc(19*var(--u));opacity:.92;margin-top:calc(2*var(--u));letter-spacing:.06em}
.b .name{background:linear-gradient(90deg,var(--blue2),var(--blue));text-align:left;border-radius:calc(8*var(--u)) 0 0 calc(8*var(--u))}
.r .name{background:linear-gradient(270deg,var(--red2),var(--red));text-align:right;border-radius:0 calc(8*var(--u)) calc(8*var(--u)) 0}
.score{width:calc(150*var(--u));display:flex;align-items:center;justify-content:center;color:#fff;font-size:calc(64*var(--u))}
.b .score{background:var(--blueL)} .r .score{background:var(--redL)}
.clock{width:calc(176*var(--u));background:#fff;color:var(--ink);display:flex;flex-direction:column;align-items:center;justify-content:center}
.clock .t{font-size:calc(56*var(--u));line-height:1}
.clock .ph{font-size:calc(16*var(--u));letter-spacing:.16em;font-weight:700;color:#5a6270;margin-top:calc(4*var(--u))}
.clock .ph.end{color:#d97706}
.side{display:flex}
.stats{position:absolute;top:calc(22*var(--u));display:flex;flex-direction:column;gap:calc(8*var(--u))}
.stats.l{left:calc(26*var(--u))} .stats.r{right:calc(26*var(--u));align-items:flex-end}
.pill{display:flex;align-items:center;gap:calc(10*var(--u));background:rgba(10,12,18,.82);color:#fff;border-radius:calc(6*var(--u));padding:calc(5*var(--u)) calc(12*var(--u));font-size:calc(18*var(--u))}
.pill b{font-size:calc(22*var(--u));min-width:calc(42*var(--u));text-align:center;border-radius:calc(4*var(--u));padding:0 calc(6*var(--u))}
.stats.l .pill b{background:var(--blue)} .stats.r .pill b{background:var(--red)}
/* split line and tiles */
.split{position:absolute;left:0;right:0;top:calc(717*var(--u));height:calc(6*var(--u));background:linear-gradient(90deg,var(--blue),#7c3aed 50%,var(--red))}
.vsplit{position:absolute;top:calc(720*var(--u));bottom:0;left:calc(957*var(--u));width:calc(6*var(--u));background:linear-gradient(180deg,#7c3aed,#3b1a66)}
.tag{position:absolute;top:calc(738*var(--u));color:#fff;font-size:calc(21*var(--u));font-weight:800;letter-spacing:.06em;padding:calc(6*var(--u)) calc(16*var(--u));border-radius:calc(6*var(--u))}
.tag.b{left:calc(20*var(--u));background:var(--blue)} .tag.r{left:calc(980*var(--u));background:var(--red)}
.tag .ours{background:#facc15;color:var(--ink);border-radius:calc(4*var(--u));padding:0 calc(8*var(--u));margin-left:calc(10*var(--u));font-size:calc(16*var(--u))}
.st{position:absolute;bottom:calc(18*var(--u));color:#fff;font-size:calc(19*var(--u));background:rgba(10,12,18,.78);padding:calc(6*var(--u)) calc(14*var(--u));border-radius:calc(6*var(--u));letter-spacing:.04em}
.st.b{left:calc(20*var(--u))} .st.r{right:calc(20*var(--u))}
.badge{position:absolute;bottom:calc(16*var(--u));left:50%;transform:translateX(-50%);background:rgba(10,12,18,.9);color:#fff;font-size:calc(24*var(--u));font-weight:700;letter-spacing:.08em;padding:calc(10*var(--u)) calc(34*var(--u));border-radius:calc(8*var(--u))}
.badge i{display:inline-block;width:calc(12*var(--u));height:calc(12*var(--u));border-radius:50%;background:#ef4444;margin-right:calc(12*var(--u));vertical-align:middle;animation:p 1.6s infinite}
@keyframes p{50%{opacity:.25}}
/* result card */
.card{position:absolute;inset:0;display:none;background:linear-gradient(90deg,#173ea8 0,#1f4fd1 50%,#d42a2a 50%,#a91d1d 100%);color:#fff}
.card.on{display:block}
.card .hd{position:absolute;top:0;left:0;right:0;height:calc(78*var(--u));background:rgba(8,10,16,.92);display:flex;align-items:center;justify-content:center;font-size:calc(28*var(--u));font-weight:700;letter-spacing:.1em}
.card .tot{position:absolute;top:calc(140*var(--u));left:50%;transform:translateX(-50%);display:flex;filter:drop-shadow(0 calc(8*var(--u)) calc(18*var(--u)) rgba(0,0,0,.4))}
.card .tot div{width:calc(300*var(--u));height:calc(190*var(--u));display:flex;flex-direction:column;align-items:center;justify-content:center}
.card .tot .lb{font-size:calc(30*var(--u));font-weight:700;letter-spacing:.06em}
.card .tot .v{font-size:calc(108*var(--u));line-height:1}
.card .tb{position:absolute;top:calc(350*var(--u));left:50%;transform:translateX(-50%);width:calc(600*var(--u));background:#fff;color:var(--ink);border-radius:calc(10*var(--u));overflow:hidden;box-shadow:0 calc(10*var(--u)) calc(26*var(--u)) rgba(0,0,0,.35)}
.card .tr{display:grid;grid-template-columns:1fr 1.3fr 1fr;align-items:center;height:calc(64*var(--u));font-size:calc(34*var(--u));border-bottom:1px solid #e3e6ec}
.card .tr:last-child{border-bottom:0}
.card .tr span{text-align:center} .card .tr .k{font-size:calc(24*var(--u));letter-spacing:.14em;font-weight:700;color:#3b4250}
.card .tr .bv{color:var(--blue2)} .card .tr .rv{color:var(--red2)}
.card .win{position:absolute;top:calc(150*var(--u));width:calc(430*var(--u));height:calc(170*var(--u));background:#facc15;color:var(--ink);display:none;align-items:center;justify-content:center;font-size:calc(66*var(--u));font-weight:900;letter-spacing:.04em;border-radius:calc(8*var(--u));line-height:1.05;text-align:center}
.card .win.l{left:calc(70*var(--u))} .card .win.r{right:calc(70*var(--u))}
.card .rbs{position:absolute;top:calc(420*var(--u));display:grid;grid-template-columns:1fr;gap:calc(14*var(--u));width:calc(430*var(--u))}
.card .rbs.l{left:calc(70*var(--u))} .card .rbs.r{right:calc(70*var(--u))}
.card .rbs div{background:rgba(255,255,255,.16);border-radius:calc(8*var(--u));padding:calc(14*var(--u)) calc(20*var(--u));font-size:calc(28*var(--u));font-weight:700;letter-spacing:.05em}
.card .wins{position:absolute;bottom:calc(70*var(--u));width:calc(430*var(--u));display:flex;justify-content:space-between;align-items:center;background:#fff;color:var(--ink);border-radius:calc(8*var(--u));padding:calc(10*var(--u)) calc(26*var(--u));font-size:calc(32*var(--u));font-weight:800;box-sizing:border-box}
.card .wins.l{left:calc(70*var(--u))} .card .wins.r{right:calc(70*var(--u))}
.card .wins b{color:#fff;background:var(--blue);padding:0 calc(18*var(--u));border-radius:calc(6*var(--u))} .card .wins.r b{background:var(--red)}
.msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#9aa3b2;font-size:calc(28*var(--u))}
</style></head><body>
<div class="stage" id="stage">
 <img class="video" id="v" src="/stream.mjpg">
 <div class="msg" id="msg">waiting for the next match ...</div>
 <div class="split"></div><div class="vsplit"></div>
 <div class="stats l" id="sl"></div><div class="stats r" id="sr"></div>
 <div class="sb">
  <div class="title" id="title">GHOSTADS LIVE</div>
  <div class="row">
   <div class="side b"><div class="name"><div class="al">BLUE ALLIANCE</div><div class="nm" id="bn">LEARNER</div><div class="rb">1 &#9733; &nbsp; 2 &nbsp; 3</div></div><div class="score num" id="bs">0</div></div>
   <div class="clock"><div class="t num" id="clk">2:30</div><div class="ph" id="ph">AUTO</div></div>
   <div class="side r"><div class="score num" id="rs">0</div><div class="name"><div class="al">RED ALLIANCE</div><div class="nm" id="rn">BASELINE</div><div class="rb">4 &nbsp; 5 &nbsp; 6</div></div></div>
  </div>
 </div>
 <div class="tag b" id="tb"></div><div class="tag r" id="tr"></div>
 <div class="st b" id="stb"></div><div class="st r" id="str"></div>
 <div class="badge"><i></i>TRAINING LIVE</div>
 <div class="card" id="card">
  <div class="hd" id="chd">MATCH RESULTS</div>
  <div class="win l" id="wl">WINNER</div><div class="win r" id="wr">WINNER</div>
  <div class="tot"><div style="background:var(--blueL)"><span class="lb" id="cbn">LEARNER</span><span class="v num" id="cbs">0</span></div>
   <div style="background:var(--redL)"><span class="lb" id="crn">BASELINE</span><span class="v num" id="crs">0</span></div></div>
  <div class="tb" id="ctb"></div>
  <div class="rbs l"><div>ROBOT 1 &#9733; OURS</div><div>ROBOT 2</div><div>ROBOT 3</div></div>
  <div class="rbs r"><div>ROBOT 4</div><div>ROBOT 5</div><div>ROBOT 6</div></div>
  <div class="wins l">WINS <b class="num" id="cwb">0</b></div><div class="wins r">WINS <b class="num" id="cwr">0</b></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id), stage=$("stage");
function fit(){stage.style.setProperty("--u",(stage.getBoundingClientRect().width/1920)+"px")}
addEventListener("resize",fit);fit();
const tg=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
addEventListener("keydown",e=>{if(e.key=="f"||e.key=="F")tg()});addEventListener("dblclick",tg);
const CAT=["LEAVE","CORAL","ALGAE","BARGE","PENALTY"];
const fmt=s=>{s=Math.max(0,Math.ceil(s));return Math.floor(s/60)+":"+String(s%60).padStart(2,"0")};
const steps=n=>n>=1e9?(n/1e9).toFixed(2)+"B":n>=1e6?(n/1e6).toFixed(1)+"M":n>=1e3?(n/1e3).toFixed(0)+"K":String(n);
function pills(el,c){el.innerHTML=["CORAL","ALGAE","BARGE"].map((k,i)=>`<div class="pill">${k}<b class="num">${Math.round(c[i+1])}</b></div>`).join("")}
function tile(tag,st,t){if(!t){tag.style.display=st.style.display="none";return}
 tag.style.display=st.style.display="";tag.innerHTML=`${t.alliance.toUpperCase()} &middot; ROBOT ${t.robot}`+(t.ours?`<span class="ours">OUR ROBOT</span>`:"");
 st.textContent=[t.goal,t.hold,t.speed.toFixed(1)+" m/s"].filter(Boolean).join("  ·  ")}
async function up(){let s;try{s=await (await fetch("/api/live")).json()}catch(e){return}
 $("title").textContent=`GHOSTADS LIVE  ·  MATCH ${s.match||"-"}  ·  TRAINING ${steps(s.step||0)} STEPS`;
 $("bn").textContent=$("cbn").textContent=s.blue_name||"LEARNER";$("rn").textContent=$("crn").textContent=s.red_name||"BASELINE";
 const c=s.card;$("card").classList.toggle("on",!!c);
 if(c){$("chd").textContent=`MATCH ${s.match} RESULTS  ·  TRAINING ${steps(s.step||0)} STEPS`;
  $("cbs").textContent=Math.round(c.scores[0]);$("crs").textContent=Math.round(c.scores[1]);
  $("ctb").innerHTML=CAT.map((k,i)=>`<div class="tr"><span class="bv num">${Math.round(c.cats[0][i])}</span><span class="k">${k}</span><span class="rv num">${Math.round(c.cats[1][i])}</span></div>`).join("");
  $("wl").style.display=c.winner==0?"flex":"none";$("wr").style.display=c.winner==1?"flex":"none";
  $("cwb").textContent=s.wins[0];$("cwr").textContent=s.wins[1];return}
 if(s.score===undefined){$("msg").style.display="flex";return}
 $("msg").style.display="none";
 $("bs").textContent=Math.round(s.score[0]);$("rs").textContent=Math.round(s.score[1]);
 $("clk").textContent=fmt(s.t_left);const ph=$("ph");ph.textContent=s.phase;ph.classList.toggle("end",s.phase=="ENDGAME");
 pills($("sl"),s.cats[0]);pills($("sr"),s.cats[1]);
 tile($("tb"),$("stb"),s.tiles&&s.tiles[0]);tile($("tr"),$("str"),s.tiles&&s.tiles[1])}
setInterval(up,100);up();
</script></body></html>"""
