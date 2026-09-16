"""Self-contained interactive dashboard for one week of predictions.

``write_dashboard(payload, path)`` renders a single HTML file with the week's
game and player predictions, a prop pricer and the model validation report
embedded as JSON. No server, no build step: open the file in a browser. The
same file publishes unchanged as a claude.ai artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = r"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F4F5F7; --surface:#FFFFFF; --surface-2:#EDEFF3; --ink:#161A21; --muted:#5F6876; --hair:#D9DDE3;
  --accent:#0F5C7A; --accent-ink:#FFFFFF; --accent-soft:#DDEBF2; --away:#C2571A; --away-soft:#F6E5D9;
  --good:#1F7A4D; --good-soft:#DDF0E5; --warn:#B7791F; --warn-soft:#F7ECD3; --bad:#B23A32; --bad-soft:#F6DDDB;
  --grid:#E6E9EE; --shadow:0 1px 2px rgba(22,26,33,.06), 0 6px 18px rgba(22,26,33,.06);
  --display:"Barlow Condensed","Arial Narrow",Impact,sans-serif; --body:"IBM Plex Sans",-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#12151A; --surface:#1A1E25; --surface-2:#232830; --ink:#E8EAEE; --muted:#98A1AE; --hair:#2A3039;
    --accent:#4FA3C7; --accent-ink:#0E1A20; --accent-soft:#1B3441; --away:#E58A4A; --away-soft:#3D2A1C;
    --good:#3FA871; --good-soft:#1C3527; --warn:#D6A03A; --warn-soft:#3A2F16; --bad:#D7605A; --bad-soft:#3E2020;
    --grid:#262C35; --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#12151A; --surface:#1A1E25; --surface-2:#232830; --ink:#E8EAEE; --muted:#98A1AE; --hair:#2A3039;
  --accent:#4FA3C7; --accent-ink:#0E1A20; --accent-soft:#1B3441; --away:#E58A4A; --away-soft:#3D2A1C;
  --good:#3FA871; --good-soft:#1C3527; --warn:#D6A03A; --warn-soft:#3A2F16; --bad:#D7605A; --bad-soft:#3E2020;
  --grid:#262C35; --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--ground);color:var(--ink)}
body{font-family:var(--body);font-size:14px;line-height:1.45;padding-inline:16px;padding-block:0 48px}
h1,h2,h3{font-family:var(--display);font-weight:600;letter-spacing:.01em;margin:0;text-wrap:balance}
h1{font-size:30px;line-height:1}
h2{font-size:22px;line-height:1.1}
h3{font-size:17px;line-height:1.2;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.num,.mono,td.num,th.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
a{color:var(--accent)}
.topbar{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;background:var(--ground);border-bottom:1px solid var(--hair);margin-inline:-16px;padding:12px 16px 0;}
.topbar-row{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px 24px;max-width:1280px;margin:0 auto}
.brand{display:flex;flex-direction:column;gap:2px}
.eyebrow{font-family:var(--display);text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:var(--muted);font-weight:600}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:12.5px;margin-left:auto}
.meta b{color:var(--ink);font-weight:500}
.tabs{display:flex;gap:2px;max-width:1280px;margin:10px auto 0;overflow-x:auto}
.tab{appearance:none;border:0;background:transparent;color:var(--muted);font-family:var(--display);font-size:17px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:8px 14px 10px;border-bottom:3px solid transparent;cursor:pointer}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}
.tab:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{max-width:1280px;margin:0 auto}
section[hidden]{display:none!important}
.panel{padding-block:20px}
.controls{display:flex;flex-wrap:wrap;gap:10px 20px;align-items:center;padding:12px 0 6px}
.control{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
.control input[type="number"],.control input[type="search"],.control select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:5px 8px;min-width:0}
.control input[type="number"]{width:86px;font-family:var(--mono)}
.control input[type="range"]{width:150px;accent-color:var(--accent)}
.chk{display:inline-flex;align-items:center;gap:5px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 14px}
.stat{background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:2px}
.stat .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:500}
.stat .v{font-family:var(--display);font-size:26px;line-height:1;font-weight:600}
.stat .s{font-size:12px;color:var(--muted)}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--hair);border-radius:8px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:13px}
th{position:sticky;top:0;background:var(--surface-2);color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.06em;font-size:11px;text-align:left;padding:8px 10px;border-bottom:1px solid var(--hair);white-space:nowrap;cursor:pointer;user-select:none}
th.num,td.num{text-align:right}
th[data-sort]:hover{color:var(--ink)}
th .arrow{font-size:9px;margin-left:4px;opacity:.7}
td{padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:middle;white-space:nowrap}
tr.row{cursor:pointer}
tr.row:hover td{background:color-mix(in srgb,var(--accent-soft) 45%,transparent)}
tr.detail td{background:var(--surface-2);white-space:normal;padding:12px 14px}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px 24px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12.5px}
.kv .k{color:var(--muted)}
.kv .v{font-family:var(--mono);text-align:right}
.team{font-family:var(--display);font-size:16px;font-weight:600;letter-spacing:.02em}
.vs{color:var(--muted);font-size:12px;margin:0 4px}
.pill{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:500;letter-spacing:.02em;white-space:nowrap}
.pill.strong{background:var(--good-soft);color:var(--good)}
.pill.lean{background:var(--warn-soft);color:var(--warn)}
.pill.pass{background:var(--surface-2);color:var(--muted)}
.pill .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.move{font-family:var(--mono);font-size:12px;color:var(--muted)}
.move.up{color:var(--good)}.move.dn{color:var(--bad)}
.plays{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin:6px 0 18px}
.play{background:var(--surface);border:1px solid var(--hair);border-left:4px solid var(--accent);border-radius:8px;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center}
.play.strong{border-left-color:var(--good)}.play.lean{border-left-color:var(--warn)}
.play .t{font-family:var(--display);font-size:18px;font-weight:600}
.play .g{color:var(--muted);font-size:12px}
.play .p{font-family:var(--mono);font-size:13px;text-align:right}
.play .stake{font-family:var(--display);font-size:20px;font-weight:600;text-align:right}
.note{color:var(--muted);font-size:12.5px;max-width:70ch}
.empty{padding:28px;text-align:center;color:var(--muted)}
.range{position:relative;height:8px;background:var(--surface-2);border-radius:4px;min-width:110px}
.range .bar{position:absolute;top:0;bottom:0;background:var(--accent-soft);border-radius:4px}
.range .mid{position:absolute;top:-2px;width:2px;height:12px;background:var(--accent)}
.props-grid{display:grid;grid-template-columns:2fr 1.4fr 1fr 1fr 1fr auto;gap:8px;align-items:center}
.props-grid input,.props-grid select{font:inherit;font-family:var(--mono);color:var(--ink);background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:6px 8px;width:100%;min-width:0}
.props-grid select{font-family:var(--body)}
.btn{appearance:none;font:inherit;font-weight:500;background:var(--accent);color:var(--accent-ink);border:0;border-radius:6px;padding:7px 12px;cursor:pointer}
.btn.ghost{background:transparent;color:var(--accent);border:1px solid var(--hair)}
.btn.ghost:hover{background:var(--accent-soft)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin:12px 0}
.chart{background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:12px 14px}
.chart h3{margin-bottom:4px}
.chart .sub{color:var(--muted);font-size:12px;margin-bottom:8px}
.chart svg{width:100%;height:auto;display:block;font-family:var(--mono);font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--muted);margin-top:6px}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.tooltip{position:fixed;pointer-events:none;background:var(--ink);color:var(--ground);padding:6px 8px;border-radius:6px;font-size:12px;font-family:var(--mono);z-index:50;display:none;white-space:nowrap}
.verdict{border:1px solid var(--hair);border-radius:8px;padding:12px 14px;background:var(--surface);margin:8px 0 12px}
.verdict .h{font-family:var(--display);font-size:20px;font-weight:600}
@media (max-width:640px){
  .props-grid{grid-template-columns:1fr 1fr}
  .stat .v{font-size:22px}
  h1{font-size:26px}
}
@media (prefers-reduced-motion: no-preference){
  tr.detail td{animation:fade .18s ease-out}
  @keyframes fade{from{opacity:.4}to{opacity:1}}
}
</style>

<div class="topbar">
  <div class="topbar-row">
    <div class="brand">
      <span class="eyebrow">NFL edge board</span>
      <h1 id="title-h1">Week __WEEK__, __SEASON__</h1>
    </div>
    <div class="meta">
      <span>generated <b id="m-generated"></b></span>
      <span>bets priced at the <b id="m-line"></b> line</span>
      <span>models <b id="m-model"></b></span>
      <span class="control">bankroll $<input id="bankroll" type="number" min="1" step="50"></span>
    </div>
  </div>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="slate" aria-selected="true">Slate</button>
    <button class="tab" role="tab" data-tab="players" aria-selected="false">Players</button>
    <button class="tab" role="tab" data-tab="props" aria-selected="false">Prop pricer</button>
    <button class="tab" role="tab" data-tab="model" aria-selected="false">Model report</button>
  </div>
</div>

<main>
<section id="tab-slate" class="panel">
  <div class="strip" id="slate-strip"></div>
  <div class="controls">
    <label class="control">min edge <input id="threshold" type="range" min="0" max="12" step="0.5"> <span class="num" id="threshold-v"></span>%</label>
    <span class="control"><label class="chk"><input type="checkbox" id="bt-spread" checked> spreads</label><label class="chk"><input type="checkbox" id="bt-total" checked> totals</label><label class="chk"><input type="checkbox" id="bt-ml" checked> moneylines</label></span>
    <span class="note">Edge is the calibrated probability minus the price's implied probability. Stakes are fractional Kelly on the bankroll above, capped per bet.</span>
  </div>
  <h3>Plays clearing the threshold</h3>
  <div class="plays" id="plays"></div>
  <h3>Full slate</h3>
  <div class="tablewrap"><table id="games-table"><thead></thead><tbody></tbody></table></div>
  <p class="note" style="margin-top:10px">Click a game for the power-model numbers, both lines, weather and rest. Spread convention: positive means the home team is favoured by that many points.</p>
</section>

<section id="tab-players" class="panel" hidden>
  <div class="controls">
    <label class="control">position <select id="pl-pos"><option value="ALL">All</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label>
    <label class="control">team <select id="pl-team"></select></label>
    <label class="control">search <input id="pl-q" type="search" placeholder="player name"></label>
    <span class="note">Mean is the model projection; the range is the 10th to 90th percentile. TD columns are the probability of at least one.</span>
  </div>
  <div class="tablewrap"><table id="players-table"><thead></thead><tbody></tbody></table></div>
</section>

<section id="tab-props" class="panel" hidden>
  <p class="note">Enter a prop line and the price you can get. Pricing uses this week's projection for that player: Normal for yardage, Poisson for touchdowns. Rows stay in this browser only.</p>
  <div id="props-rows" style="display:flex;flex-direction:column;gap:8px;margin:12px 0"></div>
  <div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn" id="props-add">Add prop</button><button class="btn ghost" id="props-clear">Clear all</button></div>
  <h3 style="margin-top:18px">Priced</h3>
  <div class="tablewrap"><table id="props-table"><thead></thead><tbody></tbody></table></div>
</section>

<section id="tab-model" class="panel">
  <div id="verdict"></div>
  <div class="charts" id="charts"></div>
  <h3>Spread model versus the opening line, out of fold</h3>
  <div class="tablewrap"><table id="roi-spread"><thead></thead><tbody></tbody></table></div>
  <h3 style="margin-top:16px">Total model versus the opening line, out of fold</h3>
  <div class="tablewrap"><table id="roi-total"><thead></thead><tbody></tbody></table></div>
  <h3 style="margin-top:16px">Player models, out of fold</h3>
  <div class="tablewrap"><table id="player-metrics"><thead></thead><tbody></tbody></table></div>
  <p class="note" style="margin-top:10px">Out-of-fold means every validation season was predicted by models trained only on earlier seasons. ROI assumes flat stakes at -110. The confidence interval is a bootstrap over the bets; "P(ROI&gt;0)" is the share of bootstrap resamples with positive return. CLV is closing-line value: how far the closing line moved toward the model's side, in points.</p>
</section>
</main>
<div class="tooltip" id="tooltip"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = (s, el=document) => el.querySelector(s);
const $$ = (s, el=document) => Array.from(el.querySelectorAll(s));
const isNum = v => typeof v === 'number' && isFinite(v);
const f = (v, d=1) => isNum(v) ? v.toFixed(d) : '–';
const sg = (v, d=1) => isNum(v) ? (v > 0 ? '+' : '') + v.toFixed(d) : '–';
const pct = (v, d=1) => isNum(v) ? (100*v).toFixed(d) + '%' : '–';
const odds = v => isNum(v) ? (v > 0 ? '+' + Math.round(v) : String(Math.round(v))) : '–';
const payout = o => o > 0 ? o/100 : 100/Math.abs(o);
const implied = o => o > 0 ? 100/(o+100) : (-o)/((-o)+100);
const kelly = (p, o, frac, cap) => { const b = payout(o); const fk = (p*b - (1-p))/b; return Math.max(0, Math.min(cap, fk*frac)); };
const erf = x => { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1/(1+0.3275911*x); const y = 1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x); return s*y; };
const ncdf = z => 0.5*(1+erf(z/Math.SQRT2));
const poisAtLeast = (lam, k) => { let p = Math.exp(-lam), cum = 0; for (let i=0;i<k;i++){ cum += p; p *= lam/(i+1); } return Math.max(0, 1-cum); };
const B = D.betting || {breakeven_110: 0.5238, kelly_fraction: 0.25, max_stake_frac: 0.03, min_edge_pct: 2};
const state = { bankroll: D.bankroll || 1000, thr: B.min_edge_pct ?? 2, bt: {spread:true,total:true,ml:true}, sort:{key:'kick',dir:1}, psort:{key:'mean',dir:-1}, open:null, pos:'ALL', team:'ALL', q:'' };
try { const s = localStorage.getItem('nfl-edge-board'); if (s) Object.assign(state, JSON.parse(s)); } catch(e){}
const save = () => { try { localStorage.setItem('nfl-edge-board', JSON.stringify({bankroll: state.bankroll, thr: state.thr, bt: state.bt})); } catch(e){} };

$('#m-generated').textContent = D.generated || '';
$('#m-line').textContent = D.bet_line || 'current';
$('#m-model').textContent = (D.games && D.games[0] && D.games[0].model_type === 'edge') ? 'two-stage edge' : 'legacy ensemble';
$('#bankroll').value = state.bankroll;
$('#threshold').value = state.thr; $('#threshold-v').textContent = f(state.thr,1);
['spread','total','ml'].forEach(k => { $('#bt-'+k).checked = state.bt[k] !== false; });

// ---------- tabs
$$('.tab').forEach(b => b.addEventListener('click', () => {
  $$('.tab').forEach(x => x.setAttribute('aria-selected', x === b ? 'true' : 'false'));
  ['slate','players','props','model'].forEach(t => { $('#tab-'+t).hidden = t !== b.dataset.tab; });
}));
$('#tab-model').hidden = true;

// ---------- picks with live recomputation
function pick(g, kind){
  const p = g[kind+'_pick_p'], o = g[kind+'_pick_odds'];
  if (!isNum(p) || !isNum(o)) return null;
  const edge = 100*(p - implied(o));
  const stake = kelly(p, o, B.kelly_fraction, B.max_stake_frac) * state.bankroll;
  const label = g[kind+'_pick'];
  const cls = edge >= 5 ? 'strong' : edge >= state.thr ? 'lean' : 'pass';
  return {kind, p, o, edge, stake, label, cls, game: g};
}
function allPicks(){
  const out = [];
  (D.games||[]).forEach(g => ['spread','total','ml'].forEach(k => { if (state.bt[k] === false) return; const pk = pick(g,k); if (pk) out.push(pk); }));
  return out;
}
const pillHtml = pk => `<span class="pill ${pk.cls}"><span class="dot"></span>${pk.cls === 'strong' ? 'strong' : pk.cls === 'lean' ? 'lean' : 'pass'}</span>`;
const moveHtml = (a, b, invertGood=false) => { if (!isNum(a) || !isNum(b)) return '<span class="move">–</span>'; const d = b - a; const c = d === 0 ? '' : (d > 0) !== invertGood ? 'up' : 'dn'; return `<span class="move ${c}">${sg(a,1)} → ${sg(b,1)}</span>`; };

function renderStrip(){
  const picks = allPicks().filter(p => p.edge >= state.thr);
  const stake = picks.reduce((s,p) => s + p.stake, 0);
  const m = (D.metrics||{}).edge_spread, roi = m && m.roi && m.roi.open_rows ? m.roi.open_rows['p>=0.54'] : null;
  const strong = picks.filter(p => p.cls === 'strong').length;
  const fcN = (D.forecasts||[]).length;
  $('#slate-strip').innerHTML = [
    ['Games', (D.games||[]).length, 'on the slate'],
    ['Plays', picks.length, `${strong} strong, ${picks.length-strong} lean`],
    ['Total stake', '$' + Math.round(stake), `${(100*stake/state.bankroll).toFixed(1)}% of bankroll`],
    ['Spread ROI vs open', roi && isNum(roi.roi) ? sg(100*roi.roi,1) + '%' : '–', roi ? `p≥0.54, n=${roi.n}, out of fold` : 'no validation report'],
    ['Weather', fcN, fcN ? 'outdoor games with a kickoff forecast' : 'climatology only'],
  ].map(([k,v,s]) => `<div class="stat"><span class="k">${k}</span><span class="v num">${v}</span><span class="s">${s}</span></div>`).join('');
}

function renderPlays(){
  const picks = allPicks().filter(p => p.edge >= state.thr).sort((a,b) => b.edge - a.edge);
  const el = $('#plays');
  if (!picks.length){ el.innerHTML = `<div class="empty">No play clears ${f(state.thr,1)}% edge. Lower the threshold to see leans.</div>`; return; }
  el.innerHTML = picks.map(pk => { const g = pk.game; const kind = {spread:'Spread', total:'Total', ml:'Moneyline'}[pk.kind];
    return `<div class="play ${pk.cls}"><div class="t">${pk.label}</div><div class="stake num">$${Math.round(pk.stake)}</div>
      <div class="g">${kind} · ${g.away_team} at ${g.home_team} · ${g.kickoff||''}</div>
      <div class="p">p ${pct(pk.p)} · ${odds(pk.o)} · edge ${sg(pk.edge,1)}%</div></div>`; }).join('');
}

const GCOLS = [
  {key:'kick', label:'Kick', get:g => (g.gameday||'').slice(5,10) + ' ' + (g.kickoff||''), cls:'mono'},
  {key:'match', label:'Matchup', get:g => g.away_team + ' at ' + g.home_team, html:g => `<span class="team">${g.away_team}</span><span class="vs">at</span><span class="team">${g.home_team}</span>`},
  {key:'spread', label:'Spread open→now', get:g => g.spread_now, html:g => moveHtml(g.spread_open, g.spread_now), num:true},
  {key:'model_margin', label:'Model margin', get:g => g.model_margin, html:g => sg(g.model_margin,1), num:true},
  {key:'spread_pick', label:'Spread pick', get:g => { const p = pick(g,'spread'); return p ? p.edge : -99; }, html:g => { const p = pick(g,'spread'); return p ? `${p.label} <span class="num">${pct(p.p,0)}</span> ${pillHtml(p)}` : '–'; }},
  {key:'total', label:'Total open→now', get:g => g.total_now, html:g => moveHtml(g.total_open, g.total_now), num:true},
  {key:'model_total', label:'Model total', get:g => g.model_total, html:g => f(g.model_total,1), num:true},
  {key:'total_pick', label:'Total pick', get:g => { const p = pick(g,'total'); return p ? p.edge : -99; }, html:g => { const p = pick(g,'total'); return p ? `${p.label} <span class="num">${pct(p.p,0)}</span> ${pillHtml(p)}` : '–'; }},
  {key:'win', label:'Home win: model / market', get:g => g.p_home_win, html:g => `<span class="num">${pct(g.p_home_win,0)}</span> <span class="num" style="color:var(--muted)">/ ${pct(g.market_p_home_win,0)}</span>`, num:true},
  {key:'ml_pick', label:'ML pick', get:g => { const p = pick(g,'ml'); return p ? p.edge : -99; }, html:g => { const p = pick(g,'ml'); return p ? `${p.label} ${odds(p.o)} ${pillHtml(p)}` : '–'; }},
];
function detailHtml(g){
  const kv = rows => `<div class="kv">${rows.map(([k,v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join('')}</div>`;
  const ps = pick(g,'spread'), pt = pick(g,'total'), pm = pick(g,'ml');
  return `<div class="detail-grid">
    <div><h3>Spread</h3>${kv([['opening line', sg(g.spread_open,1)],['current line', sg(g.spread_now,1)],['power model margin', sg(g.pf_margin,1)],['edge model margin (bet line)', sg(g.model_margin,1)],['edge vs bet line', sg(g.spread_edge_pts,1) + ' pts'],['P(home cover) at current', pct(g.p_home_cover_now ?? g.p_home_cover)],['P(home cover) at open', pct(g.p_home_cover_open)],['Normal-tail P(cover)', pct(g.p_home_cover_normal)],['stake', ps ? '$'+Math.round(ps.stake) : '–']])}</div>
    <div><h3>Total</h3>${kv([['opening total', f(g.total_open,1)],['current total', f(g.total_now,1)],['power model total', f(g.pf_total,1)],['edge model total (bet line)', f(g.model_total,1)],['edge vs bet line', sg(g.total_edge_pts,1) + ' pts'],['P(over) at current', pct(g.p_over_now ?? g.p_over)],['P(over) at open', pct(g.p_over_open)],['stake', pt ? '$'+Math.round(pt.stake) : '–']])}</div>
    <div><h3>Moneyline</h3>${kv([['model P(home win)', pct(g.p_home_win)],['market P(home win)', pct(g.market_p_home_win)],['home price', odds(g.ml_home_now)],['away price', odds(g.ml_away_now)],['home opened', odds(g.ml_h_open)],['away opened', odds(g.ml_a_open)],['stake', pm ? '$'+Math.round(pm.stake) : '–']])}</div>
    <div><h3>Context</h3>${kv([['QBs', `${g.away_qb_name||'?'} / ${g.home_qb_name||'?'}`],['stadium', g.stadium||'–'],['temp / wind', g.is_dome ? 'indoors' : `${f(g.temp_f,0)}°F / ${f(g.wind_mph,0)} mph${g.weather_imputed ? ' (climatology)' : ' (forecast)'}`],['rest (away / home)', `${f(g.away_rest,0)} / ${f(g.home_rest,0)} days`],['margin sigma', f(g.margin_sigma,1)],['total sigma', f(g.total_sigma,1)]])}</div>
  </div>`;
}
function renderGames(){
  const t = $('#games-table');
  t.querySelector('thead').innerHTML = '<tr>' + GCOLS.map(c => `<th data-sort="${c.key}" class="${c.num?'num':''}">${c.label}${state.sort.key===c.key ? `<span class="arrow">${state.sort.dir>0?'▲':'▼'}</span>` : ''}</th>`).join('') + '</tr>';
  const col = GCOLS.find(c => c.key === state.sort.key) || GCOLS[0];
  const rows = (D.games||[]).slice().sort((a,b) => { const x = col.get(a), y = col.get(b); if (x === y) return 0; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : -1) * state.sort.dir; });
  t.querySelector('tbody').innerHTML = rows.map(g => `<tr class="row" data-id="${g.game_id}">${GCOLS.map(c => `<td class="${c.num?'num':''}${c.cls?' '+c.cls:''}">${c.html ? c.html(g) : c.get(g)}</td>`).join('')}</tr>` + (state.open === g.game_id ? `<tr class="detail"><td colspan="${GCOLS.length}">${detailHtml(g)}</td></tr>` : '')).join('');
  $$('th[data-sort]', t).forEach(th => th.addEventListener('click', () => { const k = th.dataset.sort; state.sort = {key:k, dir: state.sort.key === k ? -state.sort.dir : 1}; renderGames(); }));
  $$('tr.row', t).forEach(tr => tr.addEventListener('click', () => { state.open = state.open === tr.dataset.id ? null : tr.dataset.id; renderGames(); }));
}
function renderSlate(){ renderStrip(); renderPlays(); renderGames(); }
$('#bankroll').addEventListener('input', e => { const v = parseFloat(e.target.value); if (isNum(v) && v > 0){ state.bankroll = v; save(); renderSlate(); renderProps(); } });
$('#threshold').addEventListener('input', e => { state.thr = parseFloat(e.target.value); $('#threshold-v').textContent = f(state.thr,1); save(); renderSlate(); });
['spread','total','ml'].forEach(k => $('#bt-'+k).addEventListener('change', e => { state.bt[k] = e.target.checked; save(); renderSlate(); }));

// ---------- players
const PSTATS = [['passing_yards','Pass yds'],['passing_tds','Pass TD'],['rushing_yards','Rush yds'],['rushing_tds','Rush TD'],['receiving_yards','Rec yds'],['receiving_tds','Rec TD']];
function primaryStat(p){ return p.position === 'QB' ? 'passing_yards' : p.position === 'RB' ? 'rushing_yards' : 'receiving_yards'; }
function rangeHtml(mu, q10, q90, max){ if (!isNum(mu)) return '–'; const lo = Math.max(0, q10||0), hi = q90||mu; const L = 100*lo/max, W = 100*(hi-lo)/max, M = 100*mu/max; return `<div class="range" title="10th–90th percentile"><div class="bar" style="left:${L}%;width:${W}%"></div><div class="mid" style="left:${M}%"></div></div>`; }
function renderPlayers(){
  const teams = Array.from(new Set((D.players||[]).map(p => p.team))).sort();
  const sel = $('#pl-team'); if (sel.options.length <= 1){ sel.innerHTML = '<option value="ALL">All</option>' + teams.map(t => `<option>${t}</option>`).join(''); }
  const q = state.q.toLowerCase();
  let rows = (D.players||[]).filter(p => (state.pos === 'ALL' || p.position === state.pos) && (state.team === 'ALL' || p.team === state.team) && (!q || (p.player_name||'').toLowerCase().includes(q)));
  const cols = [
    {key:'name', label:'Player', get:p => p.player_name, html:p => `<b>${p.player_name}</b> <span style="color:var(--muted)">${p.position} · ${p.team} ${p.is_home ? 'vs' : 'at'} ${p.opponent}</span>`},
    {key:'mean', label:'Primary stat', get:p => p[primaryStat(p)+'_mu'], html:p => { const s = primaryStat(p); return `<span class="num">${f(p[s+'_mu'],1)}</span> <span style="color:var(--muted);font-size:11px">${s.replace('_',' ')}</span>`; }, num:true},
    {key:'range', label:'10th–90th', get:p => p[primaryStat(p)+'_sigma'], html:p => { const s = primaryStat(p); const max = s === 'passing_yards' ? 450 : 200; return rangeHtml(p[s+'_mu'], p[s+'_q10'], p[s+'_q90'], max) + ` <span class="num" style="font-size:11px;color:var(--muted)">${f(p[s+'_q10'],0)}–${f(p[s+'_q90'],0)}</span>`; }},
    {key:'py', label:'Pass yds', get:p => p.passing_yards_mu, html:p => f(p.passing_yards_mu,0), num:true},
    {key:'ptd', label:'Pass TD ≥1', get:p => p.passing_tds_p_any, html:p => pct(p.passing_tds_p_any,0), num:true},
    {key:'ry', label:'Rush yds', get:p => p.rushing_yards_mu, html:p => f(p.rushing_yards_mu,0), num:true},
    {key:'rtd', label:'Rush TD ≥1', get:p => p.rushing_tds_p_any, html:p => pct(p.rushing_tds_p_any,0), num:true},
    {key:'cy', label:'Rec yds', get:p => p.receiving_yards_mu, html:p => f(p.receiving_yards_mu,0), num:true},
    {key:'ctd', label:'Rec TD ≥1', get:p => p.receiving_tds_p_any, html:p => pct(p.receiving_tds_p_any,0), num:true},
    {key:'snap', label:'Snap %', get:p => p.snap_pct_ewm, html:p => pct(p.snap_pct_ewm,0), num:true},
    {key:'imp', label:'Team implied', get:p => p.team_implied_pts, html:p => f(p.team_implied_pts,1), num:true},
    {key:'inj', label:'Status', get:p => p.inj_status, html:p => ['healthy','questionable','doubtful','out'][Math.round(p.inj_status||0)] || '–'},
  ];
  const col = cols.find(c => c.key === state.psort.key) || cols[1];
  rows = rows.sort((a,b) => { const x = col.get(a), y = col.get(b); if (x === y) return 0; if (x == null || Number.isNaN(x)) return 1; if (y == null || Number.isNaN(y)) return -1; return (x > y ? 1 : -1) * state.psort.dir; });
  const t = $('#players-table');
  t.querySelector('thead').innerHTML = '<tr>' + cols.map(c => `<th data-sort="${c.key}" class="${c.num?'num':''}">${c.label}${state.psort.key===c.key ? `<span class="arrow">${state.psort.dir>0?'▲':'▼'}</span>` : ''}</th>`).join('') + '</tr>';
  t.querySelector('tbody').innerHTML = rows.length ? rows.map(p => `<tr>${cols.map(c => `<td class="${c.num?'num':''}">${c.html ? c.html(p) : c.get(p)}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${cols.length}" class="empty">No players match.</td></tr>`;
  $$('th[data-sort]', t).forEach(th => th.addEventListener('click', () => { const k = th.dataset.sort; state.psort = {key:k, dir: state.psort.key === k ? -state.psort.dir : -1}; renderPlayers(); }));
}
$('#pl-pos').addEventListener('change', e => { state.pos = e.target.value; renderPlayers(); });
$('#pl-team').addEventListener('change', e => { state.team = e.target.value; renderPlayers(); });
$('#pl-q').addEventListener('input', e => { state.q = e.target.value; renderPlayers(); });

// ---------- prop pricer
let props = [];
try { props = JSON.parse(localStorage.getItem('nfl-edge-props') || 'null') || []; } catch(e){ props = []; }
if (!props.length){
  const ex = (D.players||[]).filter(p => isNum(p.receiving_yards_mu)).sort((a,b) => b.receiving_yards_mu - a.receiving_yards_mu)[0];
  const qb = (D.players||[]).filter(p => isNum(p.passing_tds_mu)).sort((a,b) => b.passing_tds_mu - a.passing_tds_mu)[0];
  if (ex) props.push({name: ex.player_name, stat: 'receiving_yards', line: Math.round(ex.receiving_yards_mu - 5) + 0.5, over: -115, under: -105, example: true});
  if (qb) props.push({name: qb.player_name, stat: 'passing_tds', line: 1.5, over: -135, under: 110, example: true});
}
const saveProps = () => { try { localStorage.setItem('nfl-edge-props', JSON.stringify(props)); } catch(e){} };
function priceProp(r){
  const p = (D.players||[]).find(x => (x.player_name||'').toLowerCase() === (r.name||'').toLowerCase());
  if (!p || !isNum(p[r.stat+'_mu'])) return null;
  const mu = isNum(p[r.stat+'_median']) ? p[r.stat+'_median'] : p[r.stat+'_mu'], sig = p[r.stat+'_sigma'];
  const line = parseFloat(r.line); if (!isNum(line)) return null;
  const tr = (D.transforms||{})[r.stat] || 'none';
  const mu_t = isNum(p[r.stat+'_mu_t']) ? p[r.stat+'_mu_t'] : mu, sig_t = isNum(p[r.stat+'_sigma_t']) ? p[r.stat+'_sigma_t'] : sig;
  const lt = tr === 'log1p' ? Math.log1p(Math.max(line, 0)) : tr === 'sqrt' ? Math.sqrt(Math.max(line, 0)) : line;
  const pOver = r.stat.endsWith('_tds') ? poisAtLeast(p[r.stat+'_mu'], Math.ceil(line + 1e-9)) : 1 - ncdf((lt - mu_t)/Math.max(sig_t, 1e-6));
  const oo = isNum(parseFloat(r.over)) ? parseFloat(r.over) : -110, uo = isNum(parseFloat(r.under)) ? parseFloat(r.under) : -110;
  const evO = pOver*payout(oo) - (1-pOver), evU = (1-pOver)*payout(uo) - pOver;
  const side = evO >= evU ? 'OVER' : 'UNDER'; const ps = side === 'OVER' ? pOver : 1-pOver; const o = side === 'OVER' ? oo : uo;
  return {player: p.player_name, team: p.team, mu, sig, pOver, side, ps, o, edge: 100*(ps - implied(o)), ev: Math.max(evO, evU), stake: kelly(ps, o, B.kelly_fraction, B.max_stake_frac)*state.bankroll};
}
function renderProps(){
  const wrap = $('#props-rows');
  wrap.innerHTML = props.map((r, i) => `<div class="props-grid" data-i="${i}">
    <input list="players-list" placeholder="player" value="${(r.name||'').replace(/"/g,'&quot;')}" data-k="name" id="prop-name-${i}">
    <select data-k="stat" id="prop-stat-${i}">${PSTATS.map(([k,l]) => `<option value="${k}" ${r.stat===k?'selected':''}>${l}</option>`).join('')}</select>
    <input type="number" step="0.5" placeholder="line" value="${r.line ?? ''}" data-k="line" id="prop-line-${i}">
    <input type="number" step="5" placeholder="over" value="${r.over ?? ''}" data-k="over" id="prop-over-${i}">
    <input type="number" step="5" placeholder="under" value="${r.under ?? ''}" data-k="under" id="prop-under-${i}">
    <button class="btn ghost" data-del="${i}" title="remove">✕</button></div>`).join('') + `<datalist id="players-list">${(D.players||[]).map(p => `<option value="${(p.player_name||'').replace(/"/g,'&quot;')}">`).join('')}</datalist>`;
  $$('.props-grid input, .props-grid select', wrap).forEach(el => el.addEventListener('input', e => { const i = +e.target.closest('.props-grid').dataset.i; props[i][e.target.dataset.k] = e.target.value; props[i].example = false; saveProps(); renderPropsTable(); }));
  $$('button[data-del]', wrap).forEach(b => b.addEventListener('click', () => { props.splice(+b.dataset.del, 1); saveProps(); renderProps(); }));
  renderPropsTable();
}
function renderPropsTable(){
  const t = $('#props-table');
  t.querySelector('thead').innerHTML = '<tr><th>Player</th><th>Stat</th><th class="num">Line</th><th class="num">Model median</th><th class="num">Spread (±1 sd)</th><th class="num">P(over)</th><th>Pick</th><th class="num">Price</th><th class="num">Edge</th><th class="num">Stake</th></tr>';
  const rows = props.map(r => ({r, x: priceProp(r)}));
  t.querySelector('tbody').innerHTML = rows.length ? rows.map(({r,x}) => x ? `<tr><td><b>${x.player}</b>${r.example ? ' <span class="pill pass">example</span>' : ''} <span style="color:var(--muted)">${x.team}</span></td><td>${(PSTATS.find(s => s[0]===r.stat)||[,r.stat])[1]}</td><td class="num">${f(parseFloat(r.line),1)}</td><td class="num">${f(x.mu,1)}</td><td class="num">${f(x.sig,1)}</td><td class="num">${pct(x.pOver)}</td><td>${x.side} <span class="pill ${x.edge>=5?'strong':x.edge>=state.thr?'lean':'pass'}"><span class="dot"></span>${sg(x.edge,1)}%</span></td><td class="num">${odds(x.o)}</td><td class="num">${sg(x.edge,1)}%</td><td class="num">$${Math.round(x.stake)}</td></tr>`
    : `<tr><td colspan="10" style="color:var(--muted)">${r.name ? `No projection for "${r.name}" on that stat this week.` : 'Enter a player name.'}</td></tr>`).join('') : '<tr><td colspan="10" class="empty">Add a prop above.</td></tr>';
}
$('#props-add').addEventListener('click', () => { props.push({name:'', stat:'receiving_yards', line:'', over:-110, under:-110}); saveProps(); renderProps(); $(`#prop-name-${props.length-1}`).focus(); });
$('#props-clear').addEventListener('click', () => { props = []; saveProps(); renderProps(); });

// ---------- model report
const tip = $('#tooltip');
function showTip(e, html){ tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }
function hideTip(){ tip.style.display = 'none'; }
function barChart(el, cats, series, opts){
  // series: [{name, color, values}] ; one y scale ; opts: {yFmt, refLine, refLabel, yMin, yMax, sub}
  const W = 520, H = 240, m = {t: 18, r: 12, b: 34, l: 44};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  let vals = series.flatMap(s => s.values).filter(isNum); if (isNum(opts.refLine)) vals.push(opts.refLine);
  let yMin = isNum(opts.yMin) ? opts.yMin : Math.min(0, ...vals), yMax = isNum(opts.yMax) ? opts.yMax : Math.max(0, ...vals);
  if (yMax === yMin) yMax = yMin + 1;
  const pad = (yMax - yMin) * 0.08; yMax += pad; if (yMin < 0) yMin -= pad;
  const y = v => m.t + ih - (v - yMin) / (yMax - yMin) * ih;
  const n = cats.length, k = series.length, gw = iw / n, bw = Math.max(4, (gw * 0.7) / k - 2);
  const ticks = 4; let g = '';
  for (let i = 0; i <= ticks; i++){ const v = yMin + (yMax - yMin) * i / ticks; g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${m.l - 6}" y="${y(v) + 3.5}" text-anchor="end" fill="var(--muted)">${opts.yFmt(v)}</text>`; }
  let bars = '';
  cats.forEach((c, i) => {
    series.forEach((s, j) => { const v = s.values[i]; if (!isNum(v)) return; const x = m.l + i * gw + (gw - (bw + 2) * k) / 2 + j * (bw + 2); const y0 = y(Math.max(0, yMin)), y1 = y(v); const top = Math.min(y0, y1), h = Math.max(1, Math.abs(y1 - y0));
      bars += `<rect x="${x}" y="${top}" width="${bw}" height="${h}" rx="2" fill="${s.color}" data-tip="${c} · ${s.name}: ${opts.yFmt(v)}${s.extra && s.extra[i] ? ' · ' + s.extra[i] : ''}"/>`;
      if (opts.labels) bars += `<text x="${x + bw / 2}" y="${(v >= 0 ? top - 4 : top + h + 11)}" text-anchor="middle" fill="var(--ink)" font-size="10">${opts.yFmt(v)}</text>`; });
    bars += `<text x="${m.l + i * gw + gw / 2}" y="${H - m.b + 16}" text-anchor="middle" fill="var(--muted)">${c}</text>`;
  });
  let ref = '';
  if (isNum(opts.refLine)) ref = `<line x1="${m.l}" x2="${W - m.r}" y1="${y(opts.refLine)}" y2="${y(opts.refLine)}" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.5"/><text x="${W - m.r}" y="${y(opts.refLine) - 4}" text-anchor="end" fill="var(--muted)" font-size="10">${opts.refLabel || ''}</text>`;
  const zero = (yMin < 0 && yMax > 0) ? `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--muted)" stroke-width="1"/>` : '';
  el.innerHTML = `<h3>${opts.title}</h3><div class="sub">${opts.sub || ''}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.title}">${g}${zero}${ref}${bars}</svg>` +
    (series.length > 1 ? `<div class="legend">${series.map(s => `<span><span class="sw" style="background:${s.color}"></span>${s.name}</span>`).join('')}</div>` : '');
  $$('rect[data-tip]', el).forEach(r => { r.addEventListener('mousemove', e => showTip(e, r.dataset.tip)); r.addEventListener('mouseleave', hideTip); });
}
function roiRows(tbl, rep){
  const t = $(tbl);
  t.querySelector('thead').innerHTML = '<tr><th>Threshold</th><th class="num">Bets</th><th class="num">Hit rate</th><th class="num">ROI</th><th class="num">95% CI</th><th class="num">P(ROI&gt;0)</th><th class="num">Units</th><th class="num">CLV pts</th><th class="num">CLV ≥ 0</th></tr>';
  if (!rep){ t.querySelector('tbody').innerHTML = '<tr><td colspan="9" class="empty">No validation report. Run train_models.py.</td></tr>'; return; }
  t.querySelector('tbody').innerHTML = Object.entries(rep).filter(([k, r]) => r.n >= 20).map(([k, r]) => { const good = isNum(r.roi) && r.roi > 0; return `<tr><td class="mono">${k.replace('p>=', 'p ≥ ')}</td><td class="num">${r.n}</td><td class="num">${pct(r.hit_rate)}</td><td class="num" style="color:${good ? 'var(--good)' : 'var(--bad)'}">${sg(100 * r.roi, 1)}%</td><td class="num">${sg(100 * r.roi_ci_low, 1)} to ${sg(100 * r.roi_ci_high, 1)}</td><td class="num">${pct(r.p_roi_positive, 0)}</td><td class="num">${sg(r.units, 1)}</td><td class="num">${sg(r.clv_pts, 2)}</td><td class="num">${pct(r.clv_nonneg_rate, 0)}</td></tr>`; }).join('');
}
function renderModel(){
  const M = D.metrics || {}; const es = M.edge_spread, et = M.edge_total;
  const be = B.breakeven_110;
  const v = $('#verdict');
  if (es && es.roi && es.roi.open_rows){
    const r = es.roi.open_rows['p>=0.54'] || es.roi.open_rows['p>=0.50'] || {};
    const rt = et && et.roi && et.roi.open_rows ? (et.roi.open_rows['p>=0.54'] || {}) : {};
    const ok = isNum(r.roi) && r.roi > 0 && r.p_roi_positive >= 0.8;
    v.innerHTML = `<div class="verdict"><div class="h">${ok ? 'Spread model shows a validated edge against opening lines' : 'Spread model does not show a reliable edge against opening lines'}</div>
      <div class="note" style="max-width:none">On ${es.n_oof_open} out-of-fold games with a true opening line (2023 onward): at p ≥ 0.54 the model bet ${r.n || 0} games, hit ${pct(r.hit_rate)} (break-even ${pct(be)}), ROI ${sg(100 * (r.roi || 0), 1)}% with a 95% interval of ${sg(100 * (r.roi_ci_low || 0), 1)} to ${sg(100 * (r.roi_ci_high || 0), 1)}%, and gained ${sg(r.clv_pts, 2)} points of closing-line value per bet.
      Totals at the same threshold: ${rt.n || 0} bets, hit ${pct(rt.hit_rate)}, ROI ${sg(100 * (rt.roi || 0), 1)}%. Model MAE against the outcome: ${f(es.mae.model_open_rows, 2)} versus ${f(es.mae.open_line_open_rows, 2)} for the opening line and ${f(es.mae.close_line_open_rows, 2)} for the closing line.</div></div>`;
  } else v.innerHTML = '<div class="verdict"><div class="h">No edge-model validation report found</div><div class="note">Run train_models.py to produce it.</div></div>';
  const ch = $('#charts'); ch.innerHTML = '';
  if (es && es.calibration_bins){
    const bins = es.calibration_bins.filter(b => b.n >= 15); const el = document.createElement('div'); el.className = 'chart'; ch.appendChild(el);
    barChart(el, bins.map(b => b.bin), [{name: 'stated probability', color: 'var(--accent)', values: bins.map(b => 100 * b.mean_p)}, {name: 'actual hit rate', color: 'var(--away)', values: bins.map(b => 100 * b.hit_rate), extra: bins.map(b => 'n=' + b.n)}],
      {title: 'Spread calibration', sub: 'confidence bin → stated versus realised cover rate, out of fold', yFmt: v => v.toFixed(0) + '%', refLine: 100 * be, refLabel: 'break-even at -110', yMin: 40, yMax: 75, labels: true});
  }
  if (es && es.by_season){
    const seasons = Object.keys(es.by_season).filter(s => (es.by_season[s].flat || {}).n >= 20); const el = document.createElement('div'); el.className = 'chart'; ch.appendChild(el);
    barChart(el, seasons, [{name: 'ROI flat-betting every game', color: 'var(--accent)', values: seasons.map(s => 100 * (es.by_season[s].flat.roi ?? NaN)), extra: seasons.map(s => 'n=' + es.by_season[s].flat.n + (es.by_season[s].open_rows ? ', open lines ' + es.by_season[s].open_rows : ', closing lines'))}],
      {title: 'Spread ROI by season', sub: 'every out-of-fold game, model side, flat stake at -110 (2023+ against opening lines)', yFmt: v => sg(v, 0) + '%', labels: true});
  }
  if (et && et.calibration_bins){
    const bins = et.calibration_bins.filter(b => b.n >= 15); const el = document.createElement('div'); el.className = 'chart'; ch.appendChild(el);
    barChart(el, bins.map(b => b.bin), [{name: 'stated probability', color: 'var(--accent)', values: bins.map(b => 100 * b.mean_p)}, {name: 'actual hit rate', color: 'var(--away)', values: bins.map(b => 100 * b.hit_rate), extra: bins.map(b => 'n=' + b.n)}],
      {title: 'Total calibration', sub: 'confidence bin → stated versus realised over/under rate, out of fold', yFmt: v => v.toFixed(0) + '%', refLine: 100 * be, refLabel: 'break-even at -110', yMin: 40, yMax: 75, labels: true});
  }
  const pm = M.players || {}; const keys = Object.keys(pm);
  if (keys.length){
    const el = document.createElement('div'); el.className = 'chart'; ch.appendChild(el);
    const lab = {passing_yards: 'pass yds', rushing_yards: 'rush yds', receiving_yards: 'rec yds', passing_tds: 'pass TD', rushing_tds: 'rush TD', receiving_tds: 'rec TD'};
    const ks = keys.filter(k => k.endsWith('_yards'));
    barChart(el, ks.map(k => lab[k] || k), [{name: 'model MAE', color: 'var(--accent)', values: ks.map(k => pm[k].oof_overall.mae)}, {name: 'player-average baseline MAE', color: 'var(--away)', values: ks.map(k => pm[k].oof_overall.baseline_ewm_mae)}],
      {title: 'Player yardage error', sub: 'mean absolute error, out of fold, lower is better', yFmt: v => v.toFixed(0), labels: true});
  }
  roiRows('#roi-spread', es && es.roi ? es.roi.open_rows : null);
  roiRows('#roi-total', et && et.roi ? et.roi.open_rows : null);
  const t = $('#player-metrics');
  t.querySelector('thead').innerHTML = '<tr><th>Target</th><th class="num">Rows</th><th class="num">Model MAE</th><th class="num">Baseline MAE</th><th class="num">Improvement</th><th class="num">80% interval coverage</th><th class="num">TD Brier</th></tr>';
  t.querySelector('tbody').innerHTML = keys.length ? keys.map(k => { const o = pm[k].oof_overall; const imp = isNum(o.baseline_ewm_mae) ? 1 - o.mae / o.baseline_ewm_mae : NaN; return `<tr><td>${k.replace('_', ' ')}</td><td class="num">${o.n}</td><td class="num">${f(o.mae, 2)}</td><td class="num">${f(o.baseline_ewm_mae, 2)}</td><td class="num" style="color:${imp > 0 ? 'var(--good)' : 'var(--bad)'}">${pct(imp, 1)}</td><td class="num">${pct(o.cover80, 1)}</td><td class="num">${f(o.td_brier, 3)}</td></tr>`; }).join('') : '<tr><td colspan="7" class="empty">No player metrics.</td></tr>';
}

renderSlate(); renderPlayers(); renderProps(); renderModel();
})();
</script>
"""


def write_dashboard(payload: dict, path: Path) -> Path:
    def _clean(o):
        if isinstance(o, float):
            return None if (o != o or o in (float("inf"), float("-inf"))) else o
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    data = json.dumps(_clean(payload), separators=(",", ":")).replace("</", "<\\/")
    html = (TEMPLATE.replace("__PAYLOAD__", data).replace("__WEEK__", str(payload.get("week", "")))
            .replace("__SEASON__", str(payload.get("season", ""))).replace("__TITLE__", f"NFL Edge Board Week {payload.get('week', '')}"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
